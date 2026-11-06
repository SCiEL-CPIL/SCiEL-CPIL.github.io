"""Program to minimize monochromatic 4-term APs in 2-coloring of [1,256]."""
import numpy as np
import time
import random

def _count_mono4ap(coloring):
    """Count monochromatic 4-term APs. coloring is 0-indexed array of 0/1."""
    n = len(coloring)
    count = 0
    for a in range(n):
        for d in range(1, (n - a) // 3 + 1):
            i0, i1, i2, i3 = a, a + d, a + 2 * d, a + 3 * d
            if i3 >= n:
                break
            c0 = coloring[i0]
            if c0 == coloring[i1] == coloring[i2] == coloring[i3]:
                count += 1
    return count

def _count_mono4ap_fast(coloring):
    """Vectorized count of monochromatic 4-term APs."""
    n = len(coloring)
    count = 0
    for d in range(1, (n + 2) // 4 + 1):
        max_a = n - 3 * d
        if max_a < 0:
            break
        a = np.arange(max_a)
        i0 = a
        i1 = a + d
        i2 = a + 2 * d
        i3 = a + 3 * d
        mono = (coloring[i0] == coloring[i1]) & (coloring[i1] == coloring[i2]) & (coloring[i2] == coloring[i3])
        count += int(np.sum(mono))
    return count

def _thue_morse(n):
    """Thue-Morse sequence of length n."""
    result = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        result[i] = result[i >> 1] ^ (i & 1)
    return result

def _delta_for_flip(coloring, pos, ap_list_by_pos):
    """Compute change in objective if we flip coloring[pos]."""
    # For each AP containing pos, check if flipping makes it mono or breaks mono
    delta = 0
    for ap_indices in ap_list_by_pos[pos]:
        # ap_indices is a tuple of 4 positions
        c0, c1, c2, c3 = [coloring[i] for i in ap_indices]
        # Currently mono?
        is_mono_now = (c0 == c1 == c2 == c3)
        # After flipping pos, what are the new values?
        new_vals = []
        for i in ap_indices:
            if i == pos:
                new_vals.append(1 - c0)
            else:
                new_vals.append(coloring[i])
        is_mono_after = (new_vals[0] == new_vals[1] == new_vals[2] == new_vals[3])
        if is_mono_now and not is_mono_after:
            delta -= 1
        elif not is_mono_now and is_mono_after:
            delta += 1
    return delta

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Precompute all 4-term APs and which positions they contain
    # ap_list_by_pos[pos] = list of AP index tuples containing pos
    ap_list_by_pos = [[] for _ in range(N)]
    all_aps = []
    for a in range(N):
        for d in range(1, (N - a) // 3 + 1):
            i3 = a + 3 * d
            if i3 >= N:
                break
            ap = (a, a + d, a + 2 * d, i3)
            all_aps.append(ap)
            for idx in ap:
                ap_list_by_pos[idx].append(ap)

    # Generate multiple initial candidates
    candidates = []
    
    # Thue-Morse
    candidates.append(_thue_morse(N).copy())
    
    # Inverse Thue-Morse
    candidates.append(1 - _thue_morse(N))
    
    # Random initializations
    rng = np.random.RandomState(42)
    for seed in range(20):
        rng = np.random.RandomState(seed * 12345 + 678)
        candidates.append(rng.randint(0, 2, size=N).astype(np.int8))
    
    # Evaluate all candidates
    best_coloring = None
    best_score = float('inf')
    for c in candidates:
        s = _count_mono4ap_fast(c)
        if s < best_score:
            best_score = s
            best_coloring = c.copy()
    
    # Simulated annealing with multiple restarts
    def sa_optimize(coloring, max_time):
        """Run simulated annealing on a copy of coloring."""
        coloring = coloring.copy()
        current_score = _count_mono4ap_fast(coloring)
        best_local = coloring.copy()
        best_local_score = current_score
        
        T = 2.0
        T_min = 0.01
        cooling = 0.9995
        rng = random.Random(hash(tuple(coloring)) % (2**31))
        
        # Precompute position ordering for each step
        positions = list(range(N))
        
        while time.time() < max_time:
            # Pick a random position
            pos = rng.choice(positions)
            delta = _delta_for_flip(coloring, pos, ap_list_by_pos)
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                coloring[pos] = 1 - coloring[pos]
                current_score += delta
                if current_score < best_local_score:
                    best_local_score = current_score
                    best_local = coloring.copy()
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        return best_local, best_local_score

    # Run SA with multiple restarts
    total_restarts = 8
    restart_time_budget = 110.0 / total_restarts
    
    for r in range(total_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Perturb the best coloring for restarts after the first
        if r == 0:
            start_c = best_coloring.copy()
        else:
            start_c = best_coloring.copy()
            # Perturb a few random positions
            rng_p = np.random.RandomState(r * 999 + 1)
            num_flips = min(10 + r * 2, 30)
            for _ in range(num_flips):
                pos = rng_p.randint(0, N)
                start_c[pos] = 1 - start_c[pos]
        
        max_time = time.time() + restart_time_budget
        if max_time > deadline - 2.0:
            max_time = deadline - 2.0
        
        local_best, local_score = sa_optimize(start_c, max_time)
        
        if local_score < best_score:
            best_score = local_score
            best_coloring = local_best.copy()
        
        # If we've found a very good solution, do a final polish
        if best_score <= 30:
            # Do a quick coordinate descent polish
            improved = True
            while improved and time.time() < deadline - 3.0:
                improved = False
                for pos in range(N):
                    if time.time() > deadline - 3.0:
                        break
                    delta = _delta_for_flip(best_coloring, pos, ap_list_by_pos)
                    if delta < 0:
                        best_coloring[pos] = 1 - best_coloring[pos]
                        best_score += delta
                        improved = True
    
    # Final coordinate descent polish
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        for pos in range(N):
            if time.time() > deadline - 1.0:
                break
            delta = _delta_for_flip(best_coloring, pos, ap_list_by_pos)
            if delta < 0:
                best_coloring[pos] = 1 - best_coloring[pos]
                best_score += delta
                improved = True
    
    # Return as list of 0/1
    return [int(x) for x in best_coloring]
