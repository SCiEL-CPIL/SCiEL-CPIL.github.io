"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

N = 256
DEADLINE = 115.0

def _count_mono4(c):
    """Count monochromatic 4-term APs in a 0/1 numpy array of length N (0-indexed)."""
    count = 0
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        if max_a <= 0:
            break
        c0 = c[0:max_a]
        c1 = c[d:d + max_a]
        c2 = c[2 * d:2 * d + max_a]
        c3 = c[3 * d:3 * d + max_a]
        mask = (c0 == c1) & (c0 == c2) & (c0 == c3)
        count += np.count_nonzero(mask)
    return count

def _thue_morse():
    """Generate Thue-Morse sequence for 0..255."""
    c = np.zeros(N, dtype=np.int8)
    for i in range(N):
        c[i] = bin(i).count('1') % 2
    return c

def _precompute_ap_membership():
    """Precompute which APs contain each position.
    Returns list of lists: ap_list[i] = list of (d, a, k) tuples where position i = a + k*d
    in AP (a, a+d, a+2d, a+3d)."""
    ap_list = [[] for _ in range(N)]
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        if max_a <= 0:
            break
        for a in range(max_a):
            for k in range(4):
                pos = a + k * d
                ap_list[pos].append((d, a, k))
    return ap_list

def _delta_flip_fast(c, i, ap_list):
    """Compute change in mono-4AP count if we flip position i, using precomputed membership."""
    delta = 0
    for (d, a, k) in ap_list[i]:
        idxs = [a, a + d, a + 2 * d, a + 3 * d]
        # Check if currently mono
        v0 = c[idxs[0]]
        v1 = c[idxs[1]]
        v2 = c[idxs[2]]
        v3 = c[idxs[3]]
        was_mono = (v0 == v1 == v2 == v3)
        
        # After flip, position i has value 1-c[i]
        new_v = 1 - c[i]
        if k == 0:
            is_mono = (new_v == v1 == v2 == v3)
        elif k == 1:
            is_mono = (v0 == new_v == v2 == v3)
        elif k == 2:
            is_mono = (v0 == v1 == new_v == v3)
        else:
            is_mono = (v0 == v1 == v2 == new_v)
        
        if was_mono and not is_mono:
            delta -= 1
        elif not was_mono and is_mono:
            delta += 1
    return delta

def _local_search_delta(c, time_limit, ap_list=None):
    """Improve coloring via single-flip local search using delta computation."""
    if ap_list is None:
        ap_list = _precompute_ap_membership()
    start = time.time()
    current = _count_mono4(c)
    improved = True
    while improved:
        improved = False
        for i in range(N):
            if time.time() - start > time_limit:
                break
            delta = _delta_flip_fast(c, i, ap_list)
            if delta < 0:
                c[i] = 1 - c[i]
                current += delta
                improved = True
        if time.time() - start > time_limit:
            break
    return c, current

def _simulated_annealing(c, time_limit, ap_list=None, T_start=5.0, T_end=0.1, alpha=0.999):
    """Simulated annealing to escape local minima."""
    if ap_list is None:
        ap_list = _precompute_ap_membership()
    start = time.time()
    current = _count_mono4(c)
    T = T_start
    rng = np.random.default_rng(123)
    best_c = c.copy()
    best_count = current
    
    while time.time() - start < time_limit:
        # Pick random position
        i = rng.integers(0, N)
        delta = _delta_flip_fast(c, i, ap_list)
        
        if delta < 0:
            # Always accept improvement
            c[i] = 1 - c[i]
            current += delta
            if current < best_count:
                best_count = current
                best_c = c.copy()
        else:
            # Accept with probability exp(-delta/T)
            if delta == 0:
                c[i] = 1 - c[i]
            else:
                if T > 0.01 and rng.random() < np.exp(-delta / T):
                    c[i] = 1 - c[i]
                    current += delta
        
        T *= alpha
        if T < T_end:
            T = T_end
    
    return best_c, best_count

def construct():
    start_time = time.time()
    ap_list = _precompute_ap_membership()
    
    # Start with Thue-Morse
    c = _thue_morse()
    best_c = c.copy()
    best_count = _count_mono4(c)
    
    # Local search from Thue-Morse using delta
    time_remaining = DEADLINE - (time.time() - start_time)
    if time_remaining > 5:
        c, cnt = _local_search_delta(c, min(30, time_remaining - 5), ap_list)
        if cnt < best_count:
            best_count = cnt
            best_c = c.copy()
    
    # Try simulated annealing from best
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 15:
        c_sa = best_c.copy()
        c_sa, cnt_sa = _simulated_annealing(c_sa, min(20, remaining - 5), ap_list)
        if cnt_sa < best_count:
            best_count = cnt_sa
            best_c = c_sa.copy()
    
    # Try random restarts if time permits
    rng = np.random.default_rng(42)
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 10:
        num_restarts = min(int(remaining // 8), 12)
        for r in range(num_restarts):
            if time.time() - start_time > DEADLINE - 5:
                break
            c_rand = rng.integers(0, 2, size=N).astype(np.int8)
            time_left = DEADLINE - (time.time() - start_time) - 5
            if time_left < 3:
                break
            # Quick local search
            c_rand, cnt_rand = _local_search_delta(c_rand, min(5, time_left), ap_list)
            if cnt_rand < best_count:
                best_count = cnt_rand
                best_c = c_rand.copy()
            # Then try SA from this result if it's close
            if time.time() - start_time < DEADLINE - 10 and cnt_rand <= best_count + 2:
                c_sa2 = c_rand.copy()
                c_sa2, cnt_sa2 = _simulated_annealing(c_sa2, min(10, DEADLINE - (time.time() - start_time) - 5), ap_list)
                if cnt_sa2 < best_count:
                    best_count = cnt_sa2
                    best_c = c_sa2.copy()
    
    return [int(x) for x in best_c]
# EVOLVE-BLOCK-END
