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
    
    # Precompute the number of APs each position belongs to
    pos_ap_counts = np.array([len(pos_to_aps[i]) for i in range(n)], dtype=np.int32)
    
    def compute_ap_sums(signs):
        ap_sums = np.zeros(num_aps, dtype=np.int32)
        for ap_id, arr in enumerate(ap_arrays):
            ap_sums[ap_id] = int(np.sum(signs[arr]))
        return ap_sums

    def build_hist(ap_sums):
        max_hist = n + 1
        hist = np.zeros(max_hist, dtype=np.int32)
        for s in ap_sums:
            hist[abs(s)] += 1
        return hist

    def get_max_from_hist(hist):
        for k in range(len(hist) - 1, -1, -1):
            if hist[k] > 0:
                return k
        return 0

    def thue_morse_signs(n):
        signs = np.zeros(n, dtype=np.int8)
        for i in range(n):
            cnt = bin(i).count('1')
            signs[i] = 1 if cnt % 2 == 0 else -1
        return signs

    def local_search(signs, max_iters, deadline):
        signs = signs.copy()
        ap_sums = compute_ap_sums(signs)
        hist = build_hist(ap_sums)
        current_max = get_max_from_hist(hist)
        best_signs = signs.copy()
        best_max = current_max
        
        temperature = 3.0
        cooling = 0.9995
        rng = random.Random()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Decide move type: 1-flip (85%) or 2-flip (15%)
            is_2flip = rng.random() < 0.15
            
            if not is_2flip:
                # 1-flip move
                # Targeted selection: 60% chance to pick from worst APs
                if rng.random() < 0.6 and current_max > 0:
                    worst_aps = np.where(np.abs(ap_sums) == current_max)[0]
                    if len(worst_aps) > 0:
                        target_ap = rng.choice(worst_aps.tolist())
                        target_pos_list = ap_indices[target_ap]
                        i = rng.choice(target_pos_list)
                    else:
                        i = rng.randint(0, n - 1)
                else:
                    i = rng.randint(0, n - 1)
                
                positions = [i]
            else:
                # 2-flip move
                i1 = rng.randint(0, n - 1)
                i2 = rng.randint(0, n - 1)
                while i2 == i1:
                    i2 = rng.randint(0, n - 1)
                positions = [i1, i2]
            
            # Compute delta for this move
            # For each position, flipping changes AP sums by -2*sign[i] for each AP containing i
            # We need to track the max change carefully
            
            # Apply flip(s) and track changes
            old_signs = [signs[p] for p in positions]
            
            # We'll compute the new max by updating affected APs
            # Collect all affected APs (with duplicates possible)
            affected_aps = []
            for p in positions:
                affected_aps.extend(pos_to_aps[p])
            
            # Remove duplicates while preserving order for efficiency
            affected_aps_unique = list(set(affected_aps))
            
            # Save old values for affected APs
            old_values = {}
            for ap_id in affected_aps_unique:
                old_values[ap_id] = ap_sums[ap_id]
            
            # Apply flips
            for p in positions:
                old_sign = signs[p]
                signs[p] = -old_sign
                for ap_id in pos_to_aps[p]:
                    ap_sums[ap_id] -= 2 * old_sign
            
            # Update histogram and compute new max
            # We need to recompute the histogram for affected APs
            # This is expensive, so let's use a different approach:
            # Track the max by scanning affected APs
            
            # Actually, let's just recompute the max from the full ap_sums array
            # but only if it's fast enough. For now, use the histogram approach
            # but only update affected entries.
            
            # Better approach: maintain a count of how many APs have each |sum| value
            # and update only the affected ones.
            
            # Let's use a simpler approach: recompute max from ap_sums
            # This is O(num_aps) which might be too slow.
            
            # Alternative: keep the histogram and update it incrementally
            for ap_id in affected_aps_unique:
                old_val = old_values[ap_id]
                old_abs = abs(old_val)
                hist[old_abs] -= 1
                new_val = ap_sums[ap_id]
                new_abs = abs(new_val)
                hist[new_abs] += 1
            
            new_max = get_max_from_hist(hist)
            delta = new_max - current_max
            
            # Acceptance criterion
            accept = False
            if delta <= 0:
                accept = True
            elif temperature > 0.01:
                if delta < 50 * temperature:
                    accept = rng.random() < np.exp(-delta / temperature)
            
            if accept:
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
            else:
                # Revert
                for p in positions:
                    old_sign = old_signs[positions.index(p)]
                    signs[p] = old_sign
                    for ap_id in pos_to_aps[p]:
                        ap_sums[ap_id] += 2 * old_sign
                
                # Revert histogram
                for ap_id in affected_aps_unique:
                    new_val = ap_sums[ap_id]
                    new_abs = abs(new_val)
                    hist[new_abs] -= 1
                    old_val = old_values[ap_id]
                    old_abs = abs(old_val)
                    hist[old_abs] += 1
            
            temperature *= cooling
        
        return best_signs, best_max

    # Multiple restarts with different initializations
    best_overall = None
    best_overall_max = n + 1
    
    # Generate diverse balanced initializations
    initial_signs_list = []
    
    # Thue-Morse
    tm = thue_morse_signs(n)
    initial_signs_list.append(tm)
    
    # Negated Thue-Morse
    initial_signs_list.append(-tm)
    
    # Random balanced sequences with seed diversity
    seeds = [42, 123, 456, 789, 1000, 2024, 3141, 2718, 1618, 9999, 5555, 7777]
    for seed in seeds:
        rng = random.Random(seed)
        signs = np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8)
        initial_signs_list.append(signs)
    
    for signs_init in initial_signs_list:
        if time.time() > deadline - 5.0:
            break
        
        signs, mx = local_search(signs_init, max_iters=50000, deadline=deadline)
        
        if mx < best_overall_max:
            best_overall_max = mx
            best_overall = signs.copy()
            if best_overall_max <= 2:
                break
    
    # Final greedy polish
    if best_overall is not None:
        signs = best_overall.copy()
        ap_sums = compute_ap_sums(signs)
        hist = build_hist(ap_sums)
        current_max = get_max_from_hist(hist)
        
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            for i in range(n):
                if time.time() > deadline - 0.5:
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
