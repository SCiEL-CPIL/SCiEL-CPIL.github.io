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

    def count_violations(ap_sums, threshold):
        # Count how many APs have |sum| > threshold
        # Using vectorization for speed
        return int(np.sum(np.abs(ap_sums) > threshold))

    def greedy_local_search(signs_init, max_iters, deadline):
        signs = signs_init.copy()
        ap_sums = compute_ap_sums(signs)
        
        best_signs = signs.copy()
        best_max = int(np.max(np.abs(ap_sums)))
        
        # We will iterate, trying to reduce the max sum by finding a flip that lowers it
        # Strategy: 
        # 1. Find current max M.
        # 2. Identify all APs with |sum| == M.
        # 3. Try flipping positions in these APs to see if it reduces M or the count of APs at M.
        
        current_max = best_max
        
        # To make this faster, we maintain the histogram of abs sums
        max_hist = n + 1
        hist = np.zeros(max_hist, dtype=np.int32)
        for s in ap_sums:
            hist[abs(s)] += 1
            
        def get_max_from_hist(h):
            for k in range(len(h) - 1, -1, -1):
                if h[k] > 0:
                    return k
            return 0

        def update_hist(ap_id, old_sum, new_sum, h):
            h[abs(old_sum)] -= 1
            h[abs(new_sum)] += 1

        current_max = get_max_from_hist(hist)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            if current_max <= 2: # Theoretical lower bound for this problem might be higher, but let's stop if very low
                break

            # Find indices of APs with max absolute sum
            # We need to check both +current_max and -current_max
            worst_aps = np.where((ap_sums == current_max) | (ap_sums == -current_max))[0]
            
            if len(worst_aps) == 0:
                # This shouldn't happen if hist is correct, but safety
                break

            # Pick a random worst AP
            target_ap = rng.choice(worst_aps.tolist())
            target_ap_id = target_ap
            target_sum = ap_sums[target_ap_id]
            
            # Try flipping each position in this AP to see if it improves the objective
            # Objective: minimize (current_max, count_of_aps_at_max)
            # Actually, simpler: just check if flipping reduces the global max.
            # If it reduces global max, take it.
            # If it doesn't reduce global max, check if it reduces the count of APs at global max.
            
            positions_in_ap = ap_indices[target_ap_id]
            
            best_flip_idx = -1
            best_flip_gain = 0 # 1 if max reduced, 0.5 if count reduced (heuristic)
            best_new_max = current_max
            best_new_count = hist[current_max]
            
            # We need to know the current count of max violations
            curr_count = hist[current_max]
            
            for pos in positions_in_ap:
                # Simulate flip
                old_sign = signs[pos]
                new_sign = -old_sign
                
                # Update ap_sums for all APs containing pos
                affected_aps = pos_to_aps[pos]
                
                # We need to track changes to hist and ap_sums
                # To avoid deep copying, we can just simulate the effect on the specific APs
                # But checking the global max requires knowing if any *other* AP becomes larger.
                # Since we are only changing APs that contain `pos`, and we are looking at the max,
                # the new max can only be:
                # 1. The old max, if no AP at old max is affected or if they don't drop below old max.
                # 2. A new max from an affected AP that grew.
                # 3. A lower max if all APs at old max drop below old max.
                
                # Let's just do the full update for the candidate flip to be safe, 
                # but only for the positions in the AP. 
                # Actually, doing a full update for every position in the AP is O(L * K) where L is AP len, K is APs per pos.
                # L can be up to 128. K can be up to 128. 
                # 128 * 128 = 16k ops per position. 
                # If AP has 10 positions, 160k ops. 
                # This might be too slow for 50k iterations.
                
                # Optimization: Only evaluate the "most promising" flips.
                # Flips that reduce the sum of the target AP are most likely to help.
                if target_sum > 0:
                    # We want to make sum smaller (more negative)
                    # Flipping a +1 to -1 reduces sum by 2.
                    # Flipping a -1 to +1 increases sum by 2 (bad).
                    candidates = [p for p in positions_in_ap if signs[p] == 1]
                else:
                    # We want to make sum larger (more positive)
                    # Flipping a -1 to +1 increases sum by 2.
                    # Flipping a +1 to -1 decreases sum by 2 (bad).
                    candidates = [p for p in positions_in_ap if signs[p] == -1]
                
                if not candidates:
                    continue
                
                # Pick a random candidate from the "good" direction
                # Or check the first one? Random is better to escape.
                pos = rng.choice(candidates)
                
                # Apply flip
                signs[pos] = new_sign
                for ap_id in affected_aps:
                    old_s = ap_sums[ap_id]
                    new_s = old_s - 2 * old_sign
                    ap_sums[ap_id] = new_s
                    update_hist(ap_id, old_s, new_s, hist)
                
                new_max = get_max_from_hist(hist)
                
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_flip_idx = pos
                    best_flip_gain = 1.0
                elif new_max == best_new_max:
                    # Check count
                    new_count = hist[new_max]
                    if new_count < best_new_count:
                        best_new_count = new_count
                        best_flip_idx = pos
                        best_flip_gain = 0.5
                        best_new_max = new_max
                
                # Revert flip for next candidate
                signs[pos] = old_sign
                for ap_id in affected_aps:
                    new_s = ap_sums[ap_id]
                    old_s = new_s + 2 * old_sign
                    ap_sums[ap_id] = old_s
                    update_hist(ap_id, new_s, old_s, hist)
            
            if best_flip_idx != -1:
                # Apply the best flip found
                old_sign = signs[best_flip_idx]
                signs[best_flip_idx] = -old_sign
                affected_aps = pos_to_aps[best_flip_idx]
                for ap_id in affected_aps:
                    old_s = ap_sums[ap_id]
                    new_s = old_s - 2 * old_sign
                    ap_sums[ap_id] = new_s
                    update_hist(ap_id, old_s, new_s, hist)
                
                current_max = get_max_from_hist(hist)
                
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
            else:
                # No improvement found in this AP. 
                # To avoid getting stuck in a local minimum where no single flip helps,
                # we can try a random flip with low probability (SA-like) or just move on.
                # Given the "Greedy" nature, let's just move on to the next iteration.
                # But if we always pick the same worst AP, we might loop.
                # We pick a random worst AP each time, so we will eventually try others.
                pass

        return best_signs, best_max

    # Multiple restarts with different initializations
    best_overall = None
    best_overall_max = n + 1
    
    # Generate diverse balanced initializations
    initial_signs_list = []
    
    def thue_morse_signs(n):
        signs = np.zeros(n, dtype=np.int8)
        for i in range(n):
            cnt = bin(i).count('1')
            signs[i] = 1 if cnt % 2 == 0 else -1
        return signs

    tm = thue_morse_signs(n)
    initial_signs_list.append(tm)
    initial_signs_list.append(-tm)
    
    # Random balanced sequences
    seeds = [42, 123, 456, 789, 1000, 2024, 3141, 2718, 1618, 9999]
    for seed in seeds:
        rng = random.Random(seed)
        signs = np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8)
        initial_signs_list.append(signs)
    
    # Run greedy local search for each initialization
    # We allocate time based on remaining time
    num_restarts = len(initial_signs_list)
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for idx, signs_init in enumerate(initial_signs_list):
        if time.time() > deadline - 2.0:
            break
        
        # Allow more iterations for later restarts? No, fixed budget per restart is safer.
        # Estimate iterations: 1 iteration ~ 50-100us? 
        # Let's just run until deadline - small_margin.
        # Actually, let's run a fixed number of iterations that fits in the time slice.
        # 55s / 12 restarts = ~4.5s per restart.
        
        signs, mx = greedy_local_search(signs_init, max_iters=100000, deadline=deadline - 1.0)
        
        if mx < best_overall_max:
            best_overall_max = mx
            best_overall = signs.copy()
            if best_overall_max <= 2:
                break
    
    # Final greedy polish on the best result found
    if best_overall is not None:
        signs = best_overall.copy()
        ap_sums = compute_ap_sums(signs)
        max_hist = n + 1
        hist = np.zeros(max_hist, dtype=np.int32)
        for s in ap_sums:
            hist[abs(s)] += 1
            
        def get_max_from_hist(h):
            for k in range(len(h) - 1, -1, -1):
                if h[k] > 0:
                    return k
            return 0

        def update_hist(ap_id, old_sum, new_sum, h):
            h[abs(old_sum)] -= 1
            h[abs(new_sum)] += 1

        current_max = get_max_from_hist(hist)
        
        # Run polish until time runs out
        while time.time() < deadline - 0.5:
            if current_max <= 2:
                break
            
            worst_aps = np.where((ap_sums == current_max) | (ap_sums == -current_max))[0]
            if len(worst_aps) == 0:
                break
                
            target_ap = rng.choice(worst_aps.tolist())
            target_ap_id = target_ap
            target_sum = ap_sums[target_ap_id]
            
            positions_in_ap = ap_indices[target_ap_id]
            
            if target_sum > 0:
                candidates = [p for p in positions_in_ap if signs[p] == 1]
            else:
                candidates = [p for p in positions_in_ap if signs[p] == -1]
                
            if not candidates:
                continue
                
            pos = rng.choice(candidates)
            
            old_sign = signs[pos]
            signs[pos] = -old_sign
            affected_aps = pos_to_aps[pos]
            for ap_id in affected_aps:
                old_s = ap_sums[ap_id]
                new_s = old_s - 2 * old_sign
                ap_sums[ap_id] = new_s
                update_hist(ap_id, old_s, new_s, hist)
            
            new_max = get_max_from_hist(hist)
            
            if new_max < current_max:
                current_max = new_max
                if current_max < best_overall_max:
                    best_overall_max = current_max
                    best_overall = signs.copy()
            else:
                # Revert
                signs[pos] = old_sign
                for ap_id in affected_aps:
                    new_s = ap_sums[ap_id]
                    old_s = new_s + 2 * old_sign
                    ap_sums[ap_id] = old_s
                    update_hist(ap_id, new_s, old_s, hist)

    if best_overall is None:
        rng = random.Random(42)
        return [rng.choice([-1, 1]) for _ in range(n)]
    
    return [int(s) for s in best_overall]
# EVOLVE-BLOCK-END
