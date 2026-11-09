import time
import random
import math

def construct():
    n = 256
    start_time = time.time()
    time_limit = 85.0  # Safety margin from 90s

    # Precompute APs: (start, step) -> list of indices
    # AP: start, start+step, start+2*step, ... < n
    aps = []
    index_to_aps = [[] for _ in range(n)]
    
    for d in range(1, n + 1):
        for start in range(d):
            idxs = list(range(start, n, d))
            if idxs:
                ap_id = len(aps)
                aps.append(idxs)
                for idx in idxs:
                    index_to_aps[idx].append(ap_id)

    num_aps = len(aps)
    
    # Initialize signs
    # Greedy initialization: assign signs one by one to minimize max discrepancy so far
    signs = [0] * n
    # partial_sums[ap_id] is the list of partial sums for that AP
    # We need to update these as we assign signs.
    # For speed, we'll just compute the cost from scratch for the greedy step initially? 
    # No, let's maintain a structure.
    
    # To make greedy fast, we can just use a simple heuristic:
    # For each index i, try +1 and -1, pick the one that results in lower max discrepancy.
    # This is O(n * num_aps * avg_ap_len) which might be too slow for 256 steps if not optimized.
    # num_aps ~ 1400. avg_ap_len ~ 50. 256 * 1400 * 50 ~ 18 million ops. Feasible in Python? Maybe.
    # Let's try a faster greedy: 
    # Actually, a random start + strong local search is often competitive.
    # But let's try a "balanced" greedy: 
    # Start with all 0s.
    # For i in 0..n-1:
    #   Calculate delta for +1 and -1 on all APs containing i.
    #   Pick sign that minimizes the increase in the global max discrepancy.
    
    # Let's stick to a robust local search with a good initialization.
    # Initialization: Random, but then immediately run a few passes of "greedy repair".
    
    def compute_max_discrepancy(signs):
        max_disc = 0
        for idxs in aps:
            curr = 0
            for idx in idxs:
                curr += signs[idx]
                if abs(curr) > max_disc:
                    max_disc = abs(curr)
        return max_disc

    def local_search(signs, rng, time_limit, start_time):
        """
        Local search that targets the worst APs.
        """
        signs = signs.copy()
        
        # Precompute partial sums for all APs
        # partial_sums[ap_id] = list of cumulative sums
        # max_sums[ap_id] = max abs value in partial_sums[ap_id]
        partial_sums = []
        max_sums = [0] * num_aps
        
        for ap_id in range(num_aps):
            idxs = aps[ap_id]
            ps = [0] * len(idxs)
            curr = 0
            for k, idx in enumerate(idxs):
                curr += signs[idx]
                ps[k] = curr
            partial_sums.append(ps)
            max_sums[ap_id] = max(abs(x) for x in ps)
        
        current_cost = max(max_sums)
        best_cost = current_cost
        best_signs = signs.copy()
        
        T = 10.0
        cooling = 0.995
        min_T = 0.1
        iter_count = 0
        no_improve_count = 0
        max_no_improve = 1000
        
        while time.time() - start_time < time_limit:
            if T < min_T:
                break
            
            # Identify worst APs
            # Find APs with max_sums == current_cost
            # To be efficient, we can just pick a random AP that is close to the max
            # Or find the exact worst.
            # Let's find the exact worst AP to target.
            worst_ap_id = -1
            worst_val = -1
            for i in range(num_aps):
                if max_sums[i] > worst_val:
                    worst_val = max_sums[i]
                    worst_ap_id = i
            
            # Get indices of the worst AP
            target_idxs = aps[worst_ap_id]
            
            # Try flipping indices in the target AP
            # Also consider indices that affect the target AP? 
            # The target AP's sum is determined by its indices.
            # Flipping an index in the target AP changes the partial sums of that AP.
            
            # Strategy: Pick a random index from the target AP and try flipping it.
            # Or try flipping all indices in the target AP one by one and pick the best improvement.
            # Since AP length can be up to 256, trying all is O(L).
            # Let's try a random subset or the single best flip.
            
            best_flip_idx = -1
            best_new_cost = current_cost + 1 # must improve
            best_new_max_sums = None
            best_new_partial_sums = None
            
            # Candidate indices: indices in the worst AP
            # We might also want to consider indices NOT in the worst AP but that
            # contribute to other high-discrepancy APs? 
            # For now, focus on the worst AP.
            
            # To speed up, we don't need to test all indices if the AP is long.
            # But max length is 256. 256 evaluations per iteration is fine.
            
            for idx in target_idxs:
                # Simulate flipping idx
                # This idx belongs to several APs.
                # We need to update partial_sums for all APs containing idx.
                # Then calculate the new global max.
                
                # Optimization: 
                # We know that flipping idx changes the partial sums of aps containing idx.
                # For APs not containing idx, max_sums remains unchanged.
                # So new_global_max = max( max(max_sums[ap] for ap not containing idx), 
                #                           max(new_max_sums[ap] for ap containing idx) )
                
                # Let's precompute the max of unaffected APs?
                # That's expensive if we do it every time.
                # However, if the worst AP is the unique maximum, and we only affect APs containing idx,
                # and the worst AP contains idx (which it does, as idx is from target_idxs),
                # then the new global max is determined by the APs containing idx.
                # Wait, other APs not containing idx might have high sums.
                
                # Let's just compute the new cost for the affected APs and take the max with the 
                # "background" max.
                
                # To make this fast, let's maintain a sorted list or just iterate.
                # Since num_aps is ~1400, iterating is fast enough if we do it smart.
                
                # Actually, let's just evaluate the flip for the specific idx and compute the cost.
                
                # Store old values
                affected_aps = index_to_aps[idx]
                old_ps_vals = {}
                old_max_vals = {}
                for ap_id in affected_aps:
                    old_ps_vals[ap_id] = partial_sums[ap_id][:] # Copy list
                    old_max_vals[ap_id] = max_sums[ap_id]
                
                # Apply flip
                signs[idx] = -signs[idx]
                
                new_global_max = 0
                # Update affected APs
                for ap_id in affected_aps:
                    idxs = aps[ap_id]
                    ps = partial_sums[ap_id]
                    curr = 0
                    ap_max = 0
                    for k, i in enumerate(idxs):
                        curr += signs[i]
                        ps[k] = curr
                        if abs(curr) > ap_max:
                            ap_max = abs(curr)
                    max_sums[ap_id] = ap_max
                    if ap_max > new_global_max:
                        new_global_max = ap_max
                
                # Check unaffected APs
                # If the previous global max was from an unaffected AP, and it's higher than new_global_max,
                # then new_global_max is that previous max.
                # We can optimize this by keeping track of the top 2 maxes?
                # For simplicity, let's just say if new_global_max < current_cost, it's an improvement.
                # But we need the actual new cost for acceptance.
                
                # A safe but slightly slower check:
                # If new_global_max < current_cost, it's definitely an improvement (since unaffected max <= current_cost).
                # If new_global_max >= current_cost, we need to check if any unaffected AP has max > new_global_max? 
                # No, if new_global_max >= current_cost, the new global max is at least new_global_max.
                # It could be higher if an unaffected AP had max > new_global_max? No, unaffected max <= current_cost <= new_global_max.
                # So if new_global_max >= current_cost, the new global max is exactly new_global_max? 
                # Wait. If current_cost was 10. Unaffected max is 9. Affected new max is 11. New global is 11.
                # If current_cost was 10. Unaffected max is 10. Affected new max is 9. New global is 10.
                # So: new_global_cost = max(new_global_max, max_unaffected).
                # We don't have max_unaffected readily available without scanning.
                
                # Let's just scan all APs to get the true new cost if new_global_max is close to current_cost?
                # Or just accept if new_global_max < current_cost.
                # This is a valid greedy step: only accept if the affected part improves the global max.
                # It might miss cases where the global max was unaffected and we need to check if it's still the max.
                # But if new_global_max < current_cost, then the new global max is <= current_cost.
                # Since we are minimizing, any reduction in the worst AP's contribution is good.
                
                candidate_cost = new_global_max
                # Note: If there are other APs with sum == current_cost that are unaffected, 
                # the true cost is still current_cost. 
                # So this "candidate_cost" is a lower bound on the true cost.
                # If candidate_cost < current_cost, it's a definite improvement.
                # If candidate_cost == current_cost, it's neutral.
                # If candidate_cost > current_cost, it's worse.
                
                if candidate_cost < best_new_cost:
                    best_new_cost = candidate_cost
                    best_flip_idx = idx
                    # Save state for this best flip
                    best_new_max_sums = max_sums[:]
                    best_new_partial_sums = [ps[:] for ps in partial_sums] # Expensive copy?
                    # We can avoid full copy if we just restore later.
                    # But we need to compare multiple flips.
                    # Let's just store the flip index and re-evaluate the best one later?
                    # Or store the diff?
                    
                # Revert flip for next candidate
                signs[idx] = -signs[idx]
                for ap_id in affected_aps:
                    max_sums[ap_id] = old_max_vals[ap_id]
                    partial_sums[ap_id] = old_ps_vals[ap_id]
                
                # Optimization: if we found a flip that reduces the cost to 0 or 1, stop early?
                # No, just keep looking for the best flip in this AP.

            # Apply the best flip found
            if best_flip_idx != -1:
                # Re-apply the best flip
                idx = best_flip_idx
                affected_aps = index_to_aps[idx]
                signs[idx] = -signs[idx]
                new_global_max = 0
                for ap_id in affected_aps:
                    idxs = aps[ap_id]
                    ps = partial_sums[ap_id]
                    curr = 0
                    ap_max = 0
                    for k, i in enumerate(idxs):
                        curr += signs[i]
                        ps[k] = curr
                        if abs(curr) > ap_max:
                            ap_max = abs(curr)
                    max_sums[ap_id] = ap_max
                    if ap_max > new_global_max:
                        new_global_max = ap_max
                
                # Determine true new cost
                # If new_global_max < current_cost, true cost is new_global_max? 
                # No, true cost is max(new_global_max, max_unaffected).
                # Since max_unaffected <= current_cost, and new_global_max < current_cost,
                # the true cost is max(new_global_max, max_unaffected).
                # It is possible that max_unaffected == current_cost, so true cost remains current_cost.
                # To be accurate, we should compute the true cost.
                
                true_new_cost = 0
                for i in range(num_aps):
                    if max_sums[i] > true_new_cost:
                        true_new_cost = max_sums[i]
                
                current_cost = true_new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_signs = signs.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # No improvement found in this worst AP
                no_improve_count += 1
                
                # If stuck, try a random flip from anywhere to escape local minimum
                if no_improve_count > 50:
                    idx = rng.randrange(n)
                    affected_aps = index_to_aps[idx]
                    old_maxs = {ap_id: max_sums[ap_id] for ap_id in affected_aps}
                    old_pss = {ap_id: partial_sums[ap_id][:] for ap_id in affected_aps}
                    
                    signs[idx] = -signs[idx]
                    for ap_id in affected_aps:
                        idxs = aps[ap_id]
                        ps = partial_sums[ap_id]
                        curr = 0
                        ap_max = 0
                        for k, i in enumerate(idxs):
                            curr += signs[i]
                            ps[k] = curr
                            if abs(curr) > ap_max:
                                ap_max = abs(curr)
                        max_sums[ap_id] = ap_max
                    
                    true_new_cost = max(max_sums)
                    
                    # Accept if better or with probability based on T
                    delta = true_new_cost - current_cost
                    if delta < 0 or (T > 0 and rng.random() < math.exp(-delta / T)):
                        current_cost = true_new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_signs = signs.copy()
                            no_improve_count = 0
                    else:
                        # Revert
                        signs[idx] = -signs[idx]
                        for ap_id in affected_aps:
                            max_sums[ap_id] = old_maxs[ap_id]
                            partial_sums[ap_id] = old_pss[ap_id]
                    
                    no_improve_count = 0 # Reset counter after a move attempt

            T *= cooling
            iter_count += 1
            
            if best_cost <= 2:
                break

        return best_signs, best_cost

    # Main execution
    rng = random.Random(42)
    
    # Initial random signs
    signs0 = [rng.choice([-1, 1]) for _ in range(n)]
    
    best_overall = signs0
    best_cost_overall = compute_max_discrepancy(signs0)
    
    # Run local search
    # We can run it once with a long budget, or multiple times with restarts.
    # Given the 85s limit, one long run with adaptive T is good.
    # Or 2-3 runs.
    
    # Let's do 3 runs, each with ~25s budget.
    num_runs = 3
    time_per_run = (time_limit - (time.time() - start_time)) / num_runs
    
    for run in range(num_runs):
        if time.time() - start_time > time_limit - 5:
            break
            
        if run == 0:
            init_signs = signs0
        else:
            # Perturb best
            init_signs = best_overall.copy()
            num_perturb = rng.randint(5, 15)
            for _ in range(num_perturb):
                idx = rng.randrange(n)
                init_signs[idx] = -init_signs[idx]
        
        res_signs, res_cost = local_search(
            init_signs,
            rng,
            time_limit - (time.time() - start_time),
            start_time
        )
        
        if res_cost < best_cost_overall:
            best_cost_overall = res_cost
            best_overall = res_signs
            
        if best_cost_overall <= 3:
            break

    return best_overall

