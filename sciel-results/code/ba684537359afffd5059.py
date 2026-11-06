import numpy as np
import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit

    # 1. Generate APs: Only APs with length >= 2 (singletons are trivial, sum = +/-1)
    # Note: The task says "including one", but a singleton sum is always +/-1.
    # If the max sum is > 1, singletons don't matter. If max sum is 1, we are done.
    # We can safely ignore singletons for optimization as long as we check the final result.
    # Actually, to be strictly correct, if we find a solution with max AP sum (len>=2) <= 1,
    # the global max is max(1, 1) = 1.
    # If we find max AP sum (len>=2) = 2, the global max is 2.
    # So minimizing max over len>=2 APs is equivalent to minimizing global max, 
    # provided we don't accidentally create a singleton issue (which is impossible, it's fixed at 1).
    # So we can just optimize over len >= 2 APs.
    
    ap_indices = []
    for start in range(1, n + 1):
        for step in range(1, n + 1):
            terms = list(range(start, n + 1, step))
            if len(terms) >= 2:
                ap_indices.append([t - 1 for t in terms])
                
    num_aps = len(ap_indices)
    
    # 2. Precompute pos_to_aps
    pos_to_aps = [[] for _ in range(n)]
    for ap_id, idx in enumerate(ap_indices):
        for i in idx:
            pos_to_aps[i].append(ap_id)
            
    # 3. Helper functions
    def compute_initial_sums(signs):
        sums = np.zeros(num_aps, dtype=np.int32)
        for ap_id, idx in enumerate(ap_indices):
            sums[ap_id] = np.sum(signs[idx])
        return sums

    def local_search(signs, max_iters, deadline, rng):
        signs = signs.copy()
        ap_sums = compute_initial_sums(signs)
        
        # Histogram for O(1) max query
        max_possible_sum = n
        hist = np.zeros(max_possible_sum + 1, dtype=np.int32)
        for s in ap_sums:
            hist[abs(s)] += 1
            
        def get_max_from_hist():
            # Scan from top down
            for k in range(max_possible_sum, -1, -1):
                if hist[k] > 0:
                    return k
            return 0
            
        current_max = get_max_from_hist()
        best_signs = signs.copy()
        best_max = current_max
        
        temperature = 2.0
        cooling = 0.999
        min_temp = 0.01
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            i = rng.randint(0, n - 1)
            old_sign = signs[i]
            new_sign = -old_sign
            delta = 2 * new_sign # -2 * old_sign
            
            affected = pos_to_aps[i]
            
            # Apply change
            for ap_id in affected:
                old_abs = abs(ap_sums[ap_id])
                hist[old_abs] -= 1
                ap_sums[ap_id] += delta
                new_abs = abs(ap_sums[ap_id])
                hist[new_abs] += 1
                
            new_max = get_max_from_hist()
            delta_max = new_max - current_max
            
            # Acceptance criteria
            if delta_max <= 0 or (temperature > min_temp and rng.random() < np.exp(-delta_max / temperature)):
                signs[i] = new_sign
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
                    ap_sums[ap_id] -= delta
                    new_abs = abs(ap_sums[ap_id])
                    hist[new_abs] += 1
                    
            temperature *= cooling
            
        return best_signs, best_max

    # 4. Main Loop with Restarts
    best_overall_signs = None
    best_overall_max = n + 1
    
    # Start with Thue-Morse (often a good starting point for discrepancy problems)
    thue_morse = np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
    best_signs, best_max = local_search(thue_morse, 5000, deadline, random.Random(42))
    best_overall_signs = best_signs
    best_overall_max = best_max
    
    # Random restarts
    seeds = [1, 2, 3, 4, 5, 10, 100, 1000, 2024, 3141, 2718, 9999, 12345, 54321, 777, 888, 999, 111, 222, 333]
    for seed in seeds:
        if time.time() > deadline - 5.0:
            break
        rng = random.Random(seed)
        signs = np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8)
        best_signs, best_max = local_search(signs, 5000, deadline, rng)
        
        if best_max < best_overall_max:
            best_overall_max = best_max
            best_overall_signs = best_signs.copy()
            
        if best_overall_max <= 2:
            # Can't get better than 2? Actually 1 is possible? 
            # Max sum 1 means all APs have sum +/-1 or 0.
            # This is very hard. Let's break if we hit 2 as it's likely optimal or near-optimal.
            # But let's try to find 1 if we have time.
            # Actually, if we break at 2, we might miss 1.
            # Let's not break, but reduce iterations if we are at 2.
            pass

    # Final polish: Greedy descent on the best solution found
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
                
                affected = pos_to_aps[i]
                
                # Apply
                for ap_id in affected:
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
                    for ap_id in affected:
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
