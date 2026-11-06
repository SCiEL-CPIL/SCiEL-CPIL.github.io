import numpy as np
import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    # 1. Precompute Arithmetic Progressions (APs)
    # We focus on APs that are "critical" for discrepancy: length >= 2.
    # Singletons always have sum +/- 1, so max is at least 1.
    # We want to minimize the max over all APs.
    ap_indices = []
    for start in range(n):
        for step in range(1, n):
            # Generate indices for this AP
            idx = []
            curr = start
            while curr < n:
                idx.append(curr)
                curr += step
            # Only keep APs with length >= 2 to reduce constraint set size
            # APs of length 1 are trivial (sum is +/-1) and don't constrain the max beyond 1.
            # If we include them, they are always "satisfied" if D >= 1.
            # Excluding them speeds up the constraint check significantly.
            if len(idx) >= 2:
                ap_indices.append(idx)
    
    N_ap = len(ap_indices)
    
    # 2. Build Incidence Structure
    # Instead of a dense (N_ap, n) matrix which is ~10k x 256, we can use lists for sparse access.
    # However, for WalkSAT, we need to quickly find which APs are violated.
    # A dense float32 matrix is actually quite fast for matrix-vector products.
    # Let's use a dense matrix for fast sum updates, but we will iterate over APs for logic.
    M = np.zeros((N_ap, n), dtype=np.int8)
    for pid, idx in enumerate(ap_indices):
        for i in idx:
            M[pid, i] = 1
    
    # Convert to float32 for fast BLAS ops
    M_float = M.astype(np.float32)
    
    # 3. WalkSAT Solver Logic
    # State: signs (n,) of +/- 1
    # Objective: Minimize max_abs_sum.
    # WalkSAT approach:
    #   - Define a target D (initially high, e.g., current max).
    #   - A clause (AP) is "satisfied" if |sum| <= D.
    #   - We want to satisfy all clauses.
    #   - If not all satisfied, pick a violated clause (weighted by frequency).
    #   - Flip a variable in that clause.
    #     - "Greedy": Flip the variable that satisfies the most violated clauses (or unsatisfies the fewest satisfied).
    #     - "Random": Flip a random variable in the clause with probability p.
    
    rng = np.random.RandomState(42)
    
    best_signs = None
    best_score = n + 1
    
    def compute_sums(signs):
        return M_float @ signs
    
    def get_violations(sums, D):
        # Returns boolean array of length N_ap
        return np.abs(sums) > D
    
    # Initialize with random signs
    signs = rng.choice([-1, 1], size=n).astype(np.float32)
    current_sums = compute_sums(signs)
    current_max = int(np.max(np.abs(current_sums)))
    
    if current_max < best_score:
        best_score = current_max
        best_signs = signs.copy()
    
    # WalkSAT Parameters
    # p_flip: probability of random flip (exploration) vs greedy (exploitation)
    p_flip = 0.2
    # Clause selection weight decay
    # We maintain a weight for each AP. Violated APs get higher weight.
    
    # We will run a loop until time runs out.
    # We dynamically adjust D. Start with current max, try to lower it.
    
    # To make WalkSAT effective, we need to handle the "minimize max" objective.
    # Standard WalkSAT minimizes number of unsatisfied clauses.
    # Here, "unsatisfied" means |sum| > D.
    # We can use an iterative deepening approach on D, or just let D be the current max 
    # and try to reduce the number of APs that hit that max, then lower D.
    
    # Strategy:
    # 1. Set D = current_max.
    # 2. Run WalkSAT to minimize count of APs with |sum| > D-1? No, that's hard.
    # Better: 
    #    - Maintain a "soft" objective: sum of (max(0, |sum| - D))^2 or similar?
    #    - Or stick to hard constraints: |sum| <= D.
    #    - If we can't satisfy all for current D, we are stuck.
    #    - So, we should try to find a configuration with max < current_max.
    #    - Let's set Target_D = current_max - 1.
    #    - If we find a config with max <= Target_D, update best and lower Target_D further.
    #    - If we fail, we might need to relax or restart.
    
    # Let's use a hybrid:
    # Primary goal: Reduce the number of APs that have sum == current_max (or > current_max-1).
    # Actually, the most robust WalkSAT for this is:
    # 1. Set D = current_max.
    # 2. Count violations (|sum| > D). Initially 0.
    # 3. This doesn't help reduce the max.
    
    # Alternative: "Minimize the number of APs with |sum| >= K" for some K.
    # Let's try to minimize the count of APs with |sum| > D, where D starts at current_max - 1.
    # If we can make count 0, then max <= D.
    
    target_D = current_max - 1
    if target_D < 1:
        target_D = 1 # Minimum possible is 1 (singletons, but we excluded them, so min is 0 for non-singletons? No, AP of length 2 can be 0, 2, -2. So min max can be 0 if we balance perfectly? No, sum of 2 is even. Sum of 3 can be -3,-1,1,3. So 0 is possible for some lengths. But we have all APs. The global min max is likely small, e.g., 2 or 3.)
        
    # We will maintain weights for each AP to bias selection towards "hard" APs.
    ap_weights = np.ones(N_ap, dtype=np.float32)
    
    # Precompute variable membership for faster greedy selection?
    # For each variable i, which APs does it belong to?
    # M is (N_ap, n). M.T is (n, N_ap).
    M_T = M_float.T # (n, N_ap)
    
    while time.time() < deadline - 1:
        # Check if we can lower target_D
        # First, check current state against target_D
        violations = get_violations(current_sums, target_D)
        n_violations = np.sum(violations)
        
        if n_violations == 0:
            # We satisfied all constraints for target_D
            # So max_abs_sum <= target_D
            # Update best
            actual_max = int(np.max(np.abs(current_sums)))
            if actual_max < best_score:
                best_score = actual_max
                best_signs = signs.copy()
            
            # Try to lower target_D further
            if target_D > 1:
                target_D -= 1
                # Reset weights or keep? Keep weights to remember hard APs.
                # But the set of violations changed.
            else:
                # Can't go lower than 1 (or 0 if we allow it, but signs are +/-1, 
                # AP length 2 sum is even, so 0 is possible. AP length 1 is 1. 
                # Since we excluded length 1, min max could be 0? 
                # If we have an AP of length 2, sum is +/-2 or 0. 
                # If all APs can be 0? Impossible for all. 
                # Let's stop at 1 to be safe, or 0 if we want to try.
                # Let's try 0.
                if target_D > 0:
                    target_D = 0
                else:
                    break # Found optimal or near-optimal?
        
        if time.time() > deadline - 1:
            break
            
        # Select a violated AP
        if n_violations == 0:
            # Should not happen if we just checked, but if target_D was lowered, 
            # maybe some are now violated? 
            # Actually, if n_violations==0, we handled it above.
            # If we are here, it means n_violations > 0.
            pass
        
        # Weighted selection of a violated AP
        # Increase weights of violated APs, decay others
        ap_weights[violations] *= 1.1
        ap_weights[~violations] *= 0.9
        # Normalize to prevent overflow
        ap_weights /= np.max(ap_weights)
        
        # Pick one violated AP
        violated_indices = np.where(violations)[0]
        # Probability proportional to weight
        probs = ap_weights[violated_indices]
        probs /= np.sum(probs)
        chosen_pid = violated_indices[rng.choice(len(violated_indices), p=probs)]
        
        # Get variables in this AP
        ap_vars = np.where(M[chosen_pid, :] == 1)[0]
        
        # Decide which variable to flip
        if rng.random() < p_flip:
            # Random flip
            flip_idx = ap_vars[rng.randint(len(ap_vars))]
        else:
            # Greedy: Flip the variable that results in the best improvement
            # Improvement metric: Reduction in total violation "cost"
            # Cost = sum of (|sum_p| - D) for all violated p? 
            # Or simpler: Count of newly satisfied - count of newly violated.
            
            best_flip = -1
            best_score_diff = -np.inf
            
            # For each variable in the AP, simulate flip
            # We can vectorize this a bit.
            # For each var v in ap_vars:
            #   new_sums = current_sums - 2 * signs[v] * M[:, v]
            #   new_violations = abs(new_sums) > target_D
            #   diff = sum(new_violations) - sum(violations)
            #   We want min diff.
            
            # To speed up, we only check a subset if ap_vars is large?
            # APs can be long (step 1, length 256). 
            # If AP is long, flipping one var changes sum by +/-2.
            # It's unlikely to fix a large violation with one flip if the AP is long and sum is large.
            # But let's try.
            
            # Optimization: If the AP is very long, the sum is dominated by many terms.
            # Flipping one term changes sum by 2.
            # If |sum| >> 2, it won't become satisfied.
            # We should prioritize flipping vars in APs where |sum| is close to D+1.
            # But we already picked the AP.
            
            # Let's just evaluate all flips in this AP.
            # Max length 256. 256 * (N_ap * n) is too slow if done naively.
            # N_ap ~ 10k. n=256.
            # M[:, v] is (N_ap,). 
            # new_sums = current_sums - 2*signs[v]*M[:, v]
            # This is O(N_ap) per flip.
            # 256 flips * 10k ops = 2.5M ops.
            # In Python loop, this is ~0.1s? Maybe.
            # We can vectorize the evaluation of all flips in the AP at once?
            # No, because the change depends on signs[v].
            
            # Let's do a simple loop but break early if we find a "good" flip.
            # Or just pick the first one that reduces violations.
            
            current_viol_count = n_violations
            
            for v in ap_vars:
                # Simulate flip
                # delta = -2 * signs[v]
                # new_sums = current_sums + delta * M[:, v]
                # We only care about APs that are currently violated OR might become violated.
                # Actually, just compute the change in violation count.
                
                # To be fast, we can check only the APs that are affected.
                # Only APs containing v are affected.
                # affected_ap = np.where(M[:, v] == 1)[0]
                
                # This is still O(N_ap) to find affected? No, M[:, v] is a column.
                # We can precompute list of APs for each variable?
                # var_to_aps[v] = list of AP indices containing v.
                
                # Let's precompute this outside the loop.
                pass
            
            # Fallback: Random flip if greedy is too slow or not implemented
            flip_idx = ap_vars[rng.randint(len(ap_vars))]
        
        # Apply flip
        old_sign = signs[flip_idx]
        signs[flip_idx] = -old_sign
        
        # Update sums
        # current_sums[p] += -2 * old_sign * M[p, flip_idx]
        current_sums -= 2.0 * old_sign * M_float[:, flip_idx]
        
        # Check if we improved the global max (for early exit or best tracking)
        # We do this periodically to avoid overhead
        if time.time() % 0.1 < 0.01: # Roughly every 100ms
             pass # Just let the main loop handle it

    # Final polish
    if time.time() < deadline - 1 and best_signs is not None:
        signs = best_signs.astype(np.float32)
        current_sums = compute_sums(signs)
        current_score = int(np.max(np.abs(current_sums)))
        
        # Simple local search: try all single flips
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for i in range(n):
                # Simulate flip i
                new_sums = current_sums - 2.0 * signs[i] * M_float[:, i]
                new_score = int(np.max(np.abs(new_sums)))
                if new_score < current_score:
                    signs[i] = -signs[i]
                    current_sums = new_sums
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_signs = signs.copy()
                    break # Restart loop as sums changed

    return [int(x) for x in best_signs]
