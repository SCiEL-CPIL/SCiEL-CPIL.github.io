import numpy as np
import time
import random
import sys

# Increase recursion limit if necessary, though iterative backtracking is safer
# But for N=256, recursion depth is 256, which is fine for default 1000.

def construct():
    start_time = time.time()
    time_limit = 55.0 # Safety margin
    
    N = 256
    
    # 1. Precompute APs
    ap_indices = []
    aps_containing = [[] for _ in range(N)]
    
    for d in range(1, N + 1):
        for s in range(1, N + 1):
            base_indices = []
            curr = s - 1
            while curr < N:
                base_indices.append(curr)
                curr += d
            
            for L in range(1, len(base_indices) + 1):
                idx_list = base_indices[:L]
                ap_idx = len(ap_indices)
                ap_indices.append(idx_list)
                for idx in idx_list:
                    aps_containing[idx].append(ap_idx)
                    
    num_aps = len(ap_indices)
    
    # Convert to lists of ints for faster access in Python loops if needed, 
    # or keep as lists.
    # For the CSP, we need to update sums for APs containing i.
    # aps_containing[i] is a list of AP indices.
    
    # We also need the length of each AP to know 'k' (unassigned count potential)
    ap_lengths = [len(lst) for lst in ap_indices]
    
    # 2. CSP Solver
    # We will use an iterative deepening or simple recursion with a time check.
    # To avoid recursion overhead, we might use a stack, but recursion is cleaner.
    # Given N=256, recursion is fine.
    
    def solve_csp(target_D, time_budget):
        """
        Returns a solution array if found within time_budget, else None.
        """
        s = [0] * N
        sums = [0] * num_aps
        # unassigned[i] is the number of unassigned variables in AP i
        unassigned = ap_lengths[:]
        
        # Precompute list of APs for each index for fast access
        # aps_containing is already a list of lists of ints
        
        def backtrack(idx):
            if time.time() - start_time > time_limit - time_budget:
                return False
            
            if idx == N:
                return True
            
            # Try +1
            s[idx] = 1
            affected_aps = aps_containing[idx]
            
            feasible = True
            # Store changes to revert if backtracking
            # We need to update sums and unassigned
            # Since we are backtracking, we can just update and then revert.
            
            for ap_idx in affected_aps:
                sums[ap_idx] += 1
                unassigned[ap_idx] -= 1
                
                # Check feasibility
                # Max possible future sum change is unassigned[ap_idx]
                # Current sum is sums[ap_idx]
                # Final sum S_f = S_cur + S_rem
                # -target_D <= S_f <= target_D
                # S_cur - unassigned <= target_D
                # S_cur + unassigned >= -target_D
                if sums[ap_idx] - unassigned[ap_idx] > target_D or sums[ap_idx] + unassigned[ap_idx] < -target_D:
                    feasible = False
                    break
            
            if feasible:
                if backtrack(idx + 1):
                    return True
            
            # Revert +1
            s[idx] = 0
            for ap_idx in affected_aps:
                sums[ap_idx] -= 1
                unassigned[ap_idx] += 1
            
            # Try -1
            s[idx] = -1
            feasible = False # Reset
            
            for ap_idx in affected_aps:
                sums[ap_idx] -= 1
                unassigned[ap_idx] -= 1
                
                if sums[ap_idx] - unassigned[ap_idx] > target_D or sums[ap_idx] + unassigned[ap_idx] < -target_D:
                    feasible = False
                    break
                else:
                    # Note: we only set feasible=False if we break. 
                    # If we don't break, it's feasible so far.
                    # But we need to check ALL.
                    pass
            
            # The logic above is slightly flawed. We must check ALL affected APs.
            # Let's rewrite the check loop properly.
            
            # Re-try with proper logic
            # We already reverted +1.
            # Now apply -1
            s[idx] = -1
            feasible = True
            for ap_idx in affected_aps:
                sums[ap_idx] -= 1
                unassigned[ap_idx] -= 1
                
                if sums[ap_idx] - unassigned[ap_idx] > target_D or sums[ap_idx] + unassigned[ap_idx] < -target_D:
                    feasible = False
                    break
            
            if feasible:
                if backtrack(idx + 1):
                    return True
            
            # Revert -1
            s[idx] = 0
            for ap_idx in affected_aps:
                sums[ap_idx] += 1
                unassigned[ap_idx] += 1
            
            return False
        
        # Start backtracking
        # We can try to optimize variable ordering, but 0..N-1 is standard.
        # Let's add a small randomization to variable order or value order?
        # No, deterministic is better for reproducibility and potential pruning structure.
        
        if backtrack(0):
            return s[:]
        else:
            return None

    # 3. Local Search Fallback
    def thue_morse(N):
        seq = np.zeros(N, dtype=np.int8)
        for i in range(N):
            parity = i.bit_count() & 1
            seq[i] = 1 if parity == 0 else -1
        return seq

    def random_init(N, rng):
        return np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.int8)

    def optimize(s_init, time_budget, rng):
        s = s_init.copy()
        sums = np.zeros(num_aps, dtype=np.int16)
        for r, idx_list in enumerate(ap_indices):
            sums[r] = np.sum(s[idx_list])
            
        current_max = int(np.max(np.abs(sums)))
        best_s = s.copy()
        best_score = current_max
        
        indices_list = list(range(N))
        
        while time.time() - start_time < time_limit - time_budget:
            rng.shuffle(indices_list)
            for i in indices_list:
                if time.time() - start_time > time_limit - time_budget:
                    break
                old_val = s[i]
                delta = -2 * old_val
                new_val = -old_val
                affected = aps_containing[i]
                
                # Check if flip improves max or keeps it same
                # Simple greedy: accept if max doesn't increase
                # We need to know if max decreases.
                # Approximation: accept if new max <= current max
                
                # Efficient check:
                # We only care if any sum exceeds current_max
                # And if we can reduce current_max.
                
                # For speed, let's just do a simple local search:
                # Accept if max_abs_sum does not increase.
                # To check this, we look at affected APs.
                # If any affected AP has |new_sum| > current_max, reject.
                # If we accept, we might lower current_max if the AP that had current_max is fixed.
                # We track current_max explicitly.
                
                # To avoid O(num_aps) check, we just check affected.
                # But we need to know if the *global* max drops.
                # This requires knowing if any *other* AP has sum == current_max.
                # If we don't track this, we might accept a move that doesn't drop the global max,
                # but that's fine, it's neutral.
                # We only reject if it *increases* the global max.
                # This happens if any affected AP has |new_sum| > current_max.
                
                reject = False
                new_max_possible = current_max
                for ap_idx in affected:
                    new_sum = sums[ap_idx] + delta
                    if abs(new_sum) > current_max:
                        reject = True
                        break
                    if abs(new_sum) > new_max_possible:
                        new_max_possible = abs(new_sum)
                
                if not reject:
                    # Apply
                    for ap_idx in affected:
                        sums[ap_idx] += delta
                    s[i] = new_val
                    
                    # Check if we can update current_max
                    # If the AP that had current_max is in affected and its sum dropped,
                    # and no other AP has current_max, we can lower it.
                    # This is complex. Let's just periodically recompute global max.
                    
                    if i % 10 == 0: # Every 10th flip
                        g_max = int(np.max(np.abs(sums)))
                        if g_max < best_score:
                            best_score = g_max
                            best_s = s.copy()
                        current_max = g_max
        
        return best_s, best_score

    # 4. Execution
    rng = random.Random(42)
    best_s_overall = None
    best_score_overall = N
    
    # Try CSP for D=3, 4, 5
    # Allocate 10s each?
    # If D=3 works, we return it immediately (best possible for N=256? maybe).
    
    # Time budget for CSP
    csp_time = 10.0
    
    for D in [3, 4, 5]:
        if time.time() - start_time > time_limit - csp_time:
            break
        sol = solve_csp(D, csp_time)
        if sol is not None:
            # Verify solution
            # Compute max discrepancy
            s_arr = np.array(sol)
            max_disc = 0
            for idx_list in ap_indices:
                s_val = abs(np.sum(s_arr[idx_list]))
                if s_val > max_disc:
                    max_disc = s_val
            
            if max_disc <= D:
                best_s_overall = s_arr
                best_score_overall = max_disc
                # If we found D=3, it's likely optimal or near optimal. 
                # We can stop and return, or try D=2?
                # D=2 is very unlikely for N=256.
                # Let's just return this if it's good.
                # But we have time, let's try to see if D=2 is possible? 
                # No, D=3 is already great.
                # Let's just do a quick local search polish to ensure it's valid and maybe find D=2?
                # Actually, if CSP found it, it's valid.
                # Let's just return it.
                return [int(x) for x in best_s_overall]

    # If CSP failed, use Local Search
    # Start with Thue-Morse
    tm = thue_morse(N)
    time_left = time_limit - (time.time() - start_time) - 1.0
    if time_left > 1.0:
        s1, score1 = optimize(tm, time_left * 0.5, rng)
        if score1 < best_score_overall:
            best_score_overall = score1
            best_s_overall = s1

    # Start with Random
    time_left = time_limit - (time.time() - start_time) - 1.0
    if time_left > 1.0:
        r1 = random_init(N, rng)
        s2, score2 = optimize(r1, time_left * 0.5, rng)
        if score2 < best_score_overall:
            best_score_overall = score2
            best_s_overall = s2

    # Final polish
    time_left = time_limit - (time.time() - start_time) - 0.5
    if time_left > 0.5 and best_s_overall is not None:
        s_final, score_final = optimize(best_s_overall, time_left * 0.9, rng)
        if score_final < best_score_overall:
            best_score_overall = score_final
            best_s_overall = s_final

    if best_s_overall is None:
        # Fallback to Thue-Morse
        best_s_overall = thue_morse(N)
        
    return [int(x) for x in best_s_overall]
