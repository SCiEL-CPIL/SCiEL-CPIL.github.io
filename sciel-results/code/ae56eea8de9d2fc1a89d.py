import numpy as np
import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s wall-clock
    
    # 1. Generate APs: Only APs with length >= 2 (singletons are trivial, sum = +/-1)
    ap_indices = []
    for start in range(1, n + 1):
        for step in range(1, n + 1):
            terms = list(range(start, n + 1, step))
            if len(terms) >= 2:
                idx = [t - 1 for t in terms]
                ap_indices.append(idx)
                
    num_aps = len(ap_indices)
    
    # 2. Precompute pos_to_aps
    pos_to_aps = [[] for _ in range(n)]
    for ap_id, idx in enumerate(ap_indices):
        for i in idx:
            pos_to_aps[i].append(ap_id)
            
    # Convert to numpy arrays for faster indexing if possible, but lists are fine for small degrees
    # Let's keep lists for flexibility in the 2-flip logic
    
    # 3. Helper functions
    def compute_initial_sums(signs):
        sums = np.zeros(num_aps, dtype=np.int32)
        for ap_id, idx in enumerate(ap_indices):
            sums[ap_id] = np.sum(signs[idx])
        return sums
        
    def get_max_abs(sums):
        return int(np.max(np.abs(sums)))

    def local_search(signs, max_iters, deadline, rng, use_2flip=True):
        signs = signs.copy()
        ap_sums = compute_initial_sums(signs)
        
        # Histogram for O(1) max update
        max_possible_sum = n
        hist = np.zeros(max_possible_sum + 1, dtype=np.int32)
        for s in ap_sums:
            hist[abs(s)] += 1
            
        def get_max_from_hist():
            for k in range(max_possible_sum, -1, -1):
                if hist[k] > 0:
                    return k
            return 0
            
        current_max = get_max_from_hist()
        best_signs = signs.copy()
        best_max = current_max
        
        temperature = 1.5
        cooling = 0.9995
        min_temp = 0.01
        
        # Precompute affected APs for each position as a list of lists for fast access
        # pos_to_aps is already a list of lists
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            # Decide move type: 1-flip or 2-flip
            # 2-flips are more expensive but escape local minima
            # Use 2-flips with lower probability, increasing as we get stuck?
            # For now, fixed probability
            is_2flip = use_2flip and (rng.random() < 0.1)
            
            if not is_2flip:
                # 1-flip move
                i = rng.randint(0, n - 1)
                old_signs = [signs[i]]
                new_signs = [-signs[i]]
                delta = 2 * new_signs[0]
                
                # Collect all affected APs (unique)
                affected_set = set(pos_to_aps[i])
                
                # Apply change
                for ap_id in affected_set:
                    old_abs = abs(ap_sums[ap_id])
                    hist[old_abs] -= 1
                    ap_sums[ap_id] += delta
                    new_abs = abs(ap_sums[ap_id])
                    hist[new_abs] += 1
                    
                new_max = get_max_from_hist()
                delta_max = new_max - current_max
                
                # Acceptance criteria
                if delta_max <= 0 or (temperature > min_temp and rng.random() < np.exp(-delta_max / temperature)):
                    signs[i] = new_signs[0]
                    current_max = new_max
                    if current_max < best_max:
                        best_max = current_max
                        best_signs = signs.copy()
                else:
                    # Revert
                    signs[i] = old_signs[0]
                    for ap_id in affected_set:
                        old_abs = abs(ap_sums[ap_id])
                        hist[old_abs] -= 1
                        ap_sums[ap_id] -= delta
                        new_abs = abs(ap_sums[ap_id])
                        hist[new_abs] += 1
                        
            else:
                # 2-flip move
                # Pick two distinct positions
                i1 = rng.randint(0, n - 1)
                i2 = rng.randint(0, n - 1)
                while i2 == i1:
                    i2 = rng.randint(0, n - 1)
                    
                old_signs = [signs[i1], signs[i2]]
                new_signs = [-old_signs[0], -old_signs[1]]
                delta1 = 2 * new_signs[0]
                delta2 = 2 * new_signs[1]
                
                # Affected APs: union of APs containing i1 and i2
                affected_set = set(pos_to_aps[i1])
                affected_set.update(pos_to_aps[i2])
                
                # Apply change
                for ap_id in affected_set:
                    old_abs = abs(ap_sums[ap_id])
                    hist[old_abs] -= 1
                    # Check if ap contains i1 and/or i2
                    # We need to know the index of i1/i2 in the AP to apply correct delta
                    # This is inefficient. We need a map from (ap_id, pos) to whether it's included?
                    # Actually, pos_to_aps[i1] gives APs containing i1.
                    # For each ap_id in affected_set:
                    #   if i1 is in ap: add delta1
                    #   if i2 is in ap: add delta2
                    
                    # To do this efficiently, we can precompute for each AP a set of indices?
                    # Or just check membership. Since APs are short on average, and affected_set is small,
                    # checking membership in ap_indices[ap_id] might be okay if we convert ap_indices to sets?
                    # No, ap_indices are lists.
                    
                    # Alternative:
                    # Precompute for each position i, a list of (ap_id, index_in_ap)
                    # Then for a 2-flip, we iterate through the list for i1 and i2.
                    # But we need to handle overlaps (APs containing both i1 and i2).
                    
                    # Let's stick to a simpler but slightly slower 2-flip implementation:
                    # We already have ap_sums.
                    # We need to update ap_sums[ap_id] by delta1 if i1 in ap, delta2 if i2 in ap.
                    
                    # Let's precompute a dictionary: pos_in_ap -> list of (ap_id, index)
                    # Actually, we can just check `if i1 in ap_indices[ap_id]`. 
                    # This is O(len(ap)) per AP.
                    # Given the small size, this is acceptable for 10% of moves.
                    
                    # Optimization: Precompute a set for each AP? No, memory.
                    # Precompute a dictionary: pos -> list of (ap_id, idx_in_ap)
                    # Let's do that once outside local_search.
                    
                    pass # Placeholder, will refactor below
            
            temperature *= cooling
            
        return best_signs, best_max

    # Refactor: Precompute pos_to_ap_indices (list of (ap_id, index_in_ap) for each pos)
    # This allows efficient delta application for 2-flips.
    pos_to_ap_info = [[] for _ in range(n)]
    for ap_id, idx in enumerate(ap_indices):
        for k, i in enumerate(idx):
            pos_to_ap_info[i].append((ap_id, k))
            
    # We still need pos_to_aps for 1-flips for simplicity, or use pos_to_ap_info.
    # Let's use pos_to_ap_info for everything.
    
    def local_search_v2(signs, max_iters, deadline, rng):
        signs = signs.copy()
        ap_sums = compute_initial_sums(signs)
        
        max_possible_sum = n
        hist = np.zeros(max_possible_sum + 1, dtype=np.int32)
        for s in ap_sums:
            hist[abs(s)] += 1
            
        def get_max_from_hist():
            for k in range(max_possible_sum, -1, -1):
                if hist[k] > 0:
                    return k
            return 0
            
        current_max = get_max_from_hist()
        best_signs = signs.copy()
        best_max = current_max
        
        temperature = 1.5
        cooling = 0.9995
        min_temp = 0.01
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            is_2flip = rng.random() < 0.2 # 20% chance of 2-flip
            
            if not is_2flip:
                # 1-flip
                i = rng.randint(0, n - 1)
                old_sign = signs[i]
                new_sign = -old_sign
                delta = 2 * new_sign
                
                affected_aps = pos_to_ap_info[i]
                # affected_aps is list of (ap_id, k). We only care about ap_id.
                # Since i appears exactly once in each AP it's in, delta is always +delta or -delta.
                
                # Apply
                for ap_id, _ in affected_aps:
                    old_abs = abs(ap_sums[ap_id])
                    hist[old_abs] -= 1
                    ap_sums[ap_id] += delta
                    new_abs = abs(ap_sums[ap_id])
                    hist[new_abs] += 1
                    
                new_max = get_max_from_hist()
                delta_max = new_max - current_max
                
                if delta_max <= 0 or (temperature > min_temp and rng.random() < np.exp(-delta_max / temperature)):
                    signs[i] = new_sign
                    current_max = new_max
                    if current_max < best_max:
                        best_max = current_max
                        best_signs = signs.copy()
                else:
                    signs[i] = old_sign
                    for ap_id, _ in affected_aps:
                        old_abs = abs(ap_sums[ap_id])
                        hist[old_abs] -= 1
                        ap_sums[ap_id] -= delta
                        new_abs = abs(ap_sums[ap_id])
                        hist[new_abs] += 1
                        
            else:
                # 2-flip
                i1 = rng.randint(0, n - 1)
                i2 = rng.randint(0, n - 1)
                while i2 == i1:
                    i2 = rng.randint(0, n - 1)
                    
                old_s1 = signs[i1]
                old_s2 = signs[i2]
                new_s1 = -old_s1
                new_s2 = -old_s2
                delta1 = 2 * new_s1
                delta2 = 2 * new_s2
                
                # Build map of ap_id -> total_delta
                # Use a dict to aggregate deltas for APs containing both i1 and i2
                ap_deltas = {}
                
                for ap_id, _ in pos_to_ap_info[i1]:
                    ap_deltas[ap_id] = ap_deltas.get(ap_id, 0) + delta1
                for ap_id, _ in pos_to_ap_info[i2]:
                    ap_deltas[ap_id] = ap_deltas.get(ap_id, 0) + delta2
                
                # Apply changes
                for ap_id, total_delta in ap_deltas.items():
                    old_abs = abs(ap_sums[ap_id])
                    hist[old_abs] -= 1
                    ap_sums[ap_id] += total_delta
                    new_abs = abs(ap_sums[ap_id])
                    hist[new_abs] += 1
                    
                new_max = get_max_from_hist()
                delta_max = new_max - current_max
                
                if delta_max <= 0 or (temperature > min_temp and rng.random() < np.exp(-delta_max / temperature)):
                    signs[i1] = new_s1
                    signs[i2] = new_s2
                    current_max = new_max
                    if current_max < best_max:
                        best_max = current_max
                        best_signs = signs.copy()
                else:
                    signs[i1] = old_s1
                    signs[i2] = old_s2
                    for ap_id, total_delta in ap_deltas.items():
                        old_abs = abs(ap_sums[ap_id])
                        hist[old_abs] -= 1
                        ap_sums[ap_id] -= total_delta
                        new_abs = abs(ap_sums[ap_id])
                        hist[new_abs] += 1
                        
            temperature *= cooling
            
        return best_signs, best_max

    # 4. Main Loop with Restarts
    best_overall_signs = None
    best_overall_max = n + 1
    
    # Start with Thue-Morse
    thue_morse = np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
    best_signs, best_max = local_search_v2(thue_morse, 20000, deadline, random.Random(42))
    best_overall_signs = best_signs
    best_overall_max = best_max
    
    # Random restarts
    seeds = [1, 2, 3, 4, 5, 10, 100, 1000, 2024, 3141, 2718, 9999, 12345, 54321, 777, 888, 999, 111, 222, 333]
    for seed in seeds:
        if time.time() > deadline - 5.0:
            break
        rng = random.Random(seed)
        signs = np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8)
        best_signs, best_max = local_search_v2(signs, 20000, deadline, rng)
        
        if best_max < best_overall_max:
            best_overall_max = best_max
            best_overall_signs = best_signs.copy()
            
        if best_overall_max <= 2:
            break
            
    # Final polish: Greedy descent (1-flip only, strict improvement) on the best solution found
    if best_overall_signs is not None:
        signs = best_overall_signs.copy()
        ap_sums = compute_initial_sums(signs)
        max_possible_sum = n
        hist = np.zeros(max_possible_sum + 1, dtype=np.int32)
        for s in ap_sums:
            hist[abs(s)] += 1
            
        def get_max_from_hist():
            for k in range(max_possible_sum, -1, -1):
                if hist[k] > 0:
                    return k
            return 0
            
        current_max = get_max_from_hist()
        
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for i in range(n):
                if time.time() > deadline - 1.0:
                    break
                old_sign = signs[i]
                new_sign = -old_sign
                delta = 2 * new_sign
                
                affected_aps = pos_to_ap_info[i]
                
                # Apply
                for ap_id, _ in affected_aps:
                    old_abs = abs(ap_sums[ap_id])
                    hist[old_abs] -= 1
                    ap_sums[ap_id] += delta
                    new_abs = abs(ap_sums[ap_id])
                    hist[new_abs] += 1
                    
                new_max = get_max_from_hist()
                
                if new_max < current_max:
                    current_max = new_max
                    signs[i] = new_sign
                    improved = True
                else:
                    # Revert
                    signs[i] = old_sign
                    for ap_id, _ in affected_aps:
                        old_abs = abs(ap_sums[ap_id])
                        hist[old_abs] -= 1
                        ap_sums[ap_id] -= delta
                        new_abs = abs(ap_sums[ap_id])
                        hist[new_abs] += 1
                        
        if current_max < best_overall_max:
            best_overall_max = current_max
            best_overall_signs = signs.copy()

    if best_overall_signs is None:
        rng = random.Random(42)
        return [rng.choice([-1, 1]) for _ in range(n)]
        
    return [int(s) for s in best_overall_signs]
