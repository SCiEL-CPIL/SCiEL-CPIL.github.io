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

def _block_coloring(block_size):
    """Generate a block coloring: 00..0 11..1 00..0 11..1 ..."""
    c = np.zeros(N, dtype=np.int8)
    for i in range(N):
        # Determine which block we are in
        block_idx = i // block_size
        if block_idx % 2 == 1:
            c[i] = 1
    return c

def _precompute_ap_membership():
    """Precompute which APs contain each position.
    Returns list of numpy arrays: ap_arrays[i] = array of (a, d, k) where position i is in AP."""
    ap_list = [[] for _ in range(N)]
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        if max_a <= 0:
            break
        for a in range(max_a):
            for k in range(4):
                pos = a + k * d
                ap_list[pos].append((a, d, k))
    ap_arrays = []
    for i in range(N):
        if ap_list[i]:
            arr = np.array(ap_list[i], dtype=np.int32)
        else:
            arr = np.empty((0, 3), dtype=np.int32)
        ap_arrays.append(arr)
    return ap_arrays

def _delta_flip_fast(c, i, ap_arrays):
    """Compute change in mono-4AP count if we flip position i, using precomputed membership."""
    ap_arr = ap_arrays[i]
    if len(ap_arr) == 0:
        return 0
    
    a_vals = ap_arr[:, 0]
    d_vals = ap_arr[:, 1]
    k_vals = ap_arr[:, 2]
    
    idx0 = a_vals
    idx1 = a_vals + d_vals
    idx2 = a_vals + 2 * d_vals
    idx3 = a_vals + 3 * d_vals
    
    v0 = c[idx0]
    v1 = c[idx1]
    v2 = c[idx2]
    v3 = c[idx3]
    
    was_mono = (v0 == v1) & (v0 == v2) & (v0 == v3)
    
    new_v = 1 - c[i]
    
    is_mono_after = np.zeros(len(ap_arr), dtype=bool)
    
    mask_k0 = (k_vals == 0)
    if np.any(mask_k0):
        is_mono_after[mask_k0] = (new_v == v1[mask_k0]) & (new_v == v2[mask_k0]) & (new_v == v3[mask_k0])
    
    mask_k1 = (k_vals == 1)
    if np.any(mask_k1):
        is_mono_after[mask_k1] = (v0[mask_k1] == new_v) & (new_v == v2[mask_k1]) & (new_v == v3[mask_k1])
    
    mask_k2 = (k_vals == 2)
    if np.any(mask_k2):
        is_mono_after[mask_k2] = (v0[mask_k2] == v1[mask_k2]) & (v1[mask_k2] == new_v) & (new_v == v3[mask_k2])
    
    mask_k3 = (k_vals == 3)
    if np.any(mask_k3):
        is_mono_after[mask_k3] = (v0[mask_k3] == v1[mask_k3]) & (v1[mask_k3] == v2[mask_k3]) & (v2[mask_k3] == new_v)
    
    delta = np.count_nonzero(~was_mono & is_mono_after) - np.count_nonzero(was_mono & ~is_mono_after)
    return int(delta)

def _local_search(c, ap_arrays, time_limit, max_passes=100):
    """Improve coloring via single-flip local search using delta computation."""
    start = time.time()
    current = _count_mono4(c)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        # Random order to break symmetry
        order = np.random.default_rng(passes * 123 + 456).permutation(N)
        for i in order:
            if time.time() - start > time_limit:
                return c, current
            delta = _delta_flip_fast(c, int(i), ap_arrays)
            if delta < 0:
                c[int(i)] = 1 - c[int(i)]
                current += delta
                improved = True
    return c, current

def _simulated_annealing(c, ap_arrays, time_limit, T_start=10.0, T_end=0.01, alpha=0.9999):
    """Simulated annealing to escape local minima."""
    start = time.time()
    current = _count_mono4(c)
    T = T_start
    rng = np.random.default_rng(789)
    best_c = c.copy()
    best_count = current
    
    # Estimate steps per second to adjust alpha? No, just use time.
    # Alpha should be tuned so T reaches T_end near time_limit.
    # ln(T_end/T_start) / ln(alpha) = steps
    # We don't know steps ahead of time.
    
    while time.time() - start < time_limit:
        i = int(rng.integers(0, N))
        delta = _delta_flip_fast(c, i, ap_arrays)
        
        if delta < 0:
            c[i] = 1 - c[i]
            current += delta
            if current < best_count:
                best_count = current
                best_c = c.copy()
        else:
            if delta == 0:
                c[i] = 1 - c[i]
            else:
                if T > 0.01:
                    # Accept with probability exp(-delta/T)
                    # If delta is large and T is small, this is rare.
                    if -delta / T > np.log(rng.random()):
                        c[i] = 1 - c[i]
                        current += delta
        
        T *= alpha
        if T < T_end:
            T = T_end
    
    return best_c, best_count

def construct():
    start_time = time.time()
    
    # Precompute AP membership for fast delta computation
    ap_arrays = _precompute_ap_membership()
    
    # Generate diverse seeds
    seeds = []
    seeds.append(_thue_morse())
    seeds.append(_block_coloring(2))
    seeds.append(_block_coloring(3))
    seeds.append(_block_coloring(4))
    seeds.append(_block_coloring(8))
    
    rng = np.random.default_rng(42)
    for _ in range(5):
        seeds.append(rng.integers(0, 2, size=N).astype(np.int8))
    
    best_c = seeds[0].copy()
    best_count = _count_mono4(best_c)
    
    # Multi-start optimization
    # We have ~110s. Let's give each seed ~10s for LS+SA.
    # 10 seeds * 10s = 100s.
    time_per_seed = 8.0
    
    for seed in seeds:
        if time.time() - start_time > DEADLINE - 5:
            break
            
        c = seed.copy()
        
        # Local Search
        t_ls = min(3.0, DEADLINE - (time.time() - start_time) - 5)
        if t_ls > 0:
            c, cnt = _local_search(c, ap_arrays, t_ls)
            if cnt < best_count:
                best_count = cnt
                best_c = c.copy()
        
        # Simulated Annealing
        t_sa = min(5.0, DEADLINE - (time.time() - start_time) - 2)
        if t_sa > 0:
            sa_c, sa_cnt = _simulated_annealing(c, ap_arrays, t_sa)
            if sa_cnt < best_count:
                best_count = sa_cnt
                best_c = sa_c.copy()
        
        # Update best from this seed's local search result if SA didn't beat it
        # (SA starts from LS result, so SA should be >= LS result usually, 
        #  but SA might find something better than the initial LS local min)
        
    # Final Polish on the global best
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 5:
        # Run a longer SA on the best solution found so far
        t_final = remaining - 2
        final_c, final_cnt = _simulated_annealing(best_c.copy(), ap_arrays, t_final, T_start=5.0, T_end=0.005, alpha=0.99995)
        if final_cnt < best_count:
            best_count = final_cnt
            best_c = final_c.copy()
            
        # Run local search again on the best
        if time.time() - start_time < DEADLINE - 1:
            t_ls_final = DEADLINE - (time.time() - start_time) - 0.5
            if t_ls_final > 0:
                ls_c, ls_cnt = _local_search(best_c.copy(), ap_arrays, t_ls_final)
                if ls_cnt < best_count:
                    best_count = ls_cnt
                    best_c = ls_c.copy()

    return [int(x) for x in best_c]
# EVOLVE-BLOCK-END
