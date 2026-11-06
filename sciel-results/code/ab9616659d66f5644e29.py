"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit

    # Precompute all arithmetic progressions of length >= 2 within [1, 256]
    ap_indices = []
    for start in range(1, n + 1):
        for step in range(1, n + 1):
            terms = list(range(start, n + 1, step))
            if len(terms) >= 2:
                ap_indices.append([t - 1 for t in terms])
    
    num_aps = len(ap_indices)
    
    # Build position-to-AP mapping
    pos_to_aps = [[] for _ in range(n)]
    for ap_id, idx in enumerate(ap_indices):
        for i in idx:
            pos_to_aps[i].append(ap_id)
    
    # Store AP indices as numpy arrays for fast summing
    ap_arrays = [np.array(idx, dtype=np.int32) for idx in ap_indices]
    
    def compute_ap_sums(signs):
        ap_sums = np.zeros(num_aps, dtype=np.int32)
        for ap_id, arr in enumerate(ap_arrays):
            ap_sums[ap_id] = int(np.sum(signs[arr]))
        return ap_sums

    def max_from_sums(ap_sums):
        return int(np.max(np.abs(ap_sums)))

    def local_search(signs, max_iters, deadline):
        signs = signs.copy()
        ap_sums = compute_ap_sums(signs)
        current_max = max_from_sums(ap_sums)
        best_signs = signs.copy()
        best_max = current_max
        
        # Track histogram of |ap_sums| for O(1) max query
        # Max possible |sum| is n, but realistically small
        max_hist = n + 1
        hist = np.zeros(max_hist, dtype=np.int32)
        for s in ap_sums:
            hist[abs(s)] += 1
        
        def get_max_from_hist(hist):
            # Find largest k with hist[k] > 0
            for k in range(max_hist - 1, -1, -1):
                if hist[k] > 0:
                    return k
            return 0
        
        current_max = get_max_from_hist(hist)
        
        temperature = 3.0
        cooling = 0.9995
        rng = random.Random()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            i = rng.randint(0, n - 1)
            old_sign = signs[i]
            
            # Flip
            signs[i] = -old_sign
            affected = pos_to_aps[i]
            
            # Update histogram and ap_sums
            for ap_id in affected:
                old_abs = abs(ap_sums[ap_id])
                hist[old_abs] -= 1
                ap_sums[ap_id] -= 2 * old_sign
                new_abs = abs(ap_sums[ap_id])
                hist[new_abs] += 1
            
            new_max = get_max_from_hist(hist)
            
            delta = new_max - current_max
            
            if delta <= 0 or (temperature > 0.01 and rng.random() < np.exp(-delta / temperature)):
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
            else:
                # Revert
                signs[i] = old_sign
                for ap_id in affected:
                    old_abs = abs(ap_sums[ap_id])
                    hist[old_abs] -= 1
                    ap_sums[ap_id] += 2 * old_sign
                    new_abs = abs(ap_sums[ap_id])
                    hist[new_abs] += 1
            
            temperature *= cooling
        
        return best_signs, best_max

    # Multiple restarts
    best_overall = None
    best_overall_max = n + 1
    
    seeds = [42, 123, 456, 789, 1000, 2024, 3141, 2718, 1618, 9999]
    for seed in seeds:
        if time.time() > deadline - 5.0:
            break
        
        rng = random.Random(seed)
        signs = np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8)
        
        signs, mx = local_search(signs, max_iters=3000, deadline=deadline)
        
        if mx < best_overall_max:
            best_overall_max = mx
            best_overall = signs.copy()
            if best_overall_max <= 2:
                break
    
    # Final greedy polish
    if best_overall is not None:
        signs = best_overall.copy()
        ap_sums = compute_ap_sums(signs)
        
        max_hist = n + 1
        hist = np.zeros(max_hist, dtype=np.int32)
        for s in ap_sums:
            hist[abs(s)] += 1
        
        def get_max_from_hist(hist):
            for k in range(max_hist - 1, -1, -1):
                if hist[k] > 0:
                    return k
            return 0
        
        current_max = get_max_from_hist(hist)
        
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for i in range(n):
                if time.time() > deadline - 1.0:
                    break
                old_sign = signs[i]
                signs[i] = -old_sign
                for ap_id in pos_to_aps[i]:
                    old_abs = abs(ap_sums[ap_id])
                    hist[old_abs] -= 1
                    ap_sums[ap_id] -= 2 * old_sign
                    new_abs = abs(ap_sums[ap_id])
                    hist[new_abs] += 1
                new_max = get_max_from_hist(hist)
                if new_max < current_max:
                    current_max = new_max
                    improved = True
                else:
                    signs[i] = old_sign
                    for ap_id in pos_to_aps[i]:
                        old_abs = abs(ap_sums[ap_id])
                        hist[old_abs] -= 1
                        ap_sums[ap_id] += 2 * old_sign
                        new_abs = abs(ap_sums[ap_id])
                        hist[new_abs] += 1
        
        if current_max < best_overall_max:
            best_overall_max = current_max
            best_overall = signs.copy()
    
    if best_overall is None:
        rng = random.Random(42)
        return [rng.choice([-1, 1]) for _ in range(n)]
    
    return [int(s) for s in best_overall]
# EVOLVE-BLOCK-END
