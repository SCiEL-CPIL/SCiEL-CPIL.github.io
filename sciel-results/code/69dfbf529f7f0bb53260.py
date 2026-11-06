import time
import random
import numpy as np
import math

def construct():
    start_time = time.time()
    time_limit = 58.0  # Safety margin for 60s limit (evaluator kills at 90s, but we should be safe)
    
    N = 256
    
    # 1. Precompute APs
    ap_indices = []
    for d in range(1, N // 2 + 1):
        for start in range(1, N - d + 1):
            indices = list(range(start - 1, N, d))
            if len(indices) >= 1:
                ap_indices.append(indices)
                
    num_aps = len(ap_indices)
    
    # Reverse mapping: for each position i in 0..N-1, which APs contain it?
    aps_containing = [[] for _ in range(N)]
    for r, indices in enumerate(ap_indices):
        for c in indices:
            aps_containing[c].append(r)
            
    # Convert to numpy arrays for faster indexing in the loop
    aps_containing_np = [np.array(aps, dtype=np.int32) for aps in aps_containing]

    def thue_morse(N):
        seq = []
        for i in range(N):
            count = bin(i).count('1')
            if count % 2 == 0:
                seq.append(1)
            else:
                seq.append(-1)
        return np.array(seq, dtype=np.int8)

    def evaluate_sequence(s):
        sums = np.zeros(num_aps, dtype=np.int32)
        for r, indices in enumerate(ap_indices):
            sums[r] = np.sum(s[indices])
        return int(np.max(np.abs(sums)))

    def run_simulated_annealing(s_init, time_budget):
        s = s_init.copy()
        
        # Initialize sums
        sums = np.zeros(num_aps, dtype=np.int32)
        for r, indices in enumerate(ap_indices):
            sums[r] = np.sum(s[indices])
            
        # Initialize histogram of absolute sums
        # Max possible sum is N (256). So hist size 257 is enough? 
        # Wait, sum of 256 terms of +/-1 is between -256 and 256.
        max_possible_sum = N
        hist = np.zeros(max_possible_sum + 1, dtype=np.int32)
        for val in sums:
            hist[abs(val)] += 1
            
        current_max = int(np.max(np.abs(sums)))
        while hist[current_max] == 0:
            current_max -= 1
            
        best_s = s.copy()
        best_score = current_max
        
        rng = random.Random(42)
        
        # SA Parameters
        T_start = 5.0
        T_end = 0.01
        # We will run for a fixed number of steps or until time is up.
        # Let's estimate steps per second.
        
        elapsed = 0.0
        steps = 0
        last_check = time.time()
        
        # Adaptive cooling
        # We want to finish cooling by the end of the time budget.
        # But we don't know the exact steps we can do.
        # Let's use a simple linear cooling based on time.
        # Or just a fixed number of iterations if we can estimate speed.
        # Let's do a quick benchmark of 1000 steps to estimate speed.
        
        # Actually, let's just run until time limit, using a cooling schedule based on elapsed time.
        
        while time.time() - start_time < time_limit:
            # Determine temperature based on elapsed time in this run
            elapsed = time.time() - (start_time + (time_limit - time_budget)) # This is tricky if we reuse start_time
            # Better: pass start_time for this run? 
            # Let's just use a fixed number of steps per "chunk" and adjust T.
            
            # Simpler: Just use a fixed T decay per step, but ensure we do enough steps.
            # Let's run for a specific number of steps, say 500,000, and see how long it takes.
            # If it's too slow, we reduce steps.
            
            # Let's try a dynamic approach:
            # We have `time_budget` left.
            # We don't know steps/sec. 
            # Let's just run a loop and break on time.
            
            for _ in range(1000): # Batch of 1000 steps to reduce overhead
                if time.time() - start_time > time_limit:
                    break
                
                # Pick random index
                i = rng.randint(0, N - 1)
                
                old_val = s[i]
                delta_val = -2 * old_val
                affected_aps = aps_containing_np[i]
                
                if len(affected_aps) == 0:
                    continue
                
                # Save old sums for affected APs
                old_sums = sums[affected_aps].copy()
                
                # Apply flip
                new_sums = old_sums + delta_val
                sums[affected_aps] = new_sums
                s[i] = -old_val
                
                # Update histogram
                # We need to update hist for the changed APs
                for r in affected_aps:
                    old_abs = abs(old_sums[r])
                    new_abs = abs(new_sums[r])
                    hist[old_abs] -= 1
                    hist[new_abs] += 1
                    
                # Update current_max
                # Check if any new_abs > current_max
                max_new_abs = int(np.max(new_sums[np.abs(new_sums) > current_max])) if np.any(np.abs(new_sums) > current_max) else 0
                # This is inefficient. 
                
                # Better: 
                # If we increased the max, current_max increases.
                # If we decreased the max, current_max might decrease.
                
                # Let's just recalculate max from hist if necessary?
                # Or track it.
                
                # Efficient max update:
                # 1. If any new_abs > current_max: current_max = max(new_abs)
                # 2. Else if hist[current_max] == 0: while current_max > 0 and hist[current_max] == 0: current_max -= 1
                
                # To check if any new_abs > current_max:
                # We can just take max of new_abs values.
                new_max_candidate = int(np.max(np.abs(new_sums)))
                
                if new_max_candidate > current_max:
                    current_max = new_max_candidate
                elif hist[current_max] == 0:
                    # The old max might be gone
                    # We need to find the new max
                    # Since we only changed a few APs, the max can only drop to the next highest non-zero hist bin.
                    # But we don't know which bin.
                    # However, the max can only drop if the APs that were at max are now lower.
                    # So we just scan down from current_max.
                    while current_max > 0 and hist[current_max] == 0:
                        current_max -= 1
                        
                # Accept/Reject
                # SA Criterion
                if current_max < best_score:
                    best_score = current_max
                    best_s = s.copy()
                    
                # Temperature
                # We need a cooling schedule.
                # Let's use a simple exponential decay based on steps.
                # But we don't have a total steps count.
                # Let's use time-based cooling.
                # T = T_start * (T_end/T_start)^(elapsed / total_time)
                
                # This requires knowing total_time.
                # Let's just use a fixed T that decays slowly.
                # Or better, run multiple restarts with shorter budgets.
                
                # Let's restart the SA loop logic to be cleaner.
                
                # Actually, the inner loop is getting messy with the histogram updates in Python.
                # The `for r in affected_aps` loop is Python-level.
                # `affected_aps` can be up to ~100-200 elements.
                # 1000 steps * 100 updates = 100,000 Python iterations.
                # This is likely too slow.
                
                # Optimization: Vectorize the hist update?
                # `old_abs = np.abs(old_sums)`
                # `new_abs = np.abs(new_sums)`
                # `hist[old_abs] -= 1` -> This is tricky with numpy advanced indexing if there are duplicates in `old_abs`.
                # `np.bincount` can help.
                
                # Let's try:
                # `counts_old = np.bincount(old_abs, minlength=max_possible_sum+1)`
                # `counts_new = np.bincount(new_abs, minlength=max_possible_sum+1)`
                # `hist -= counts_old`
                # `hist += counts_new`
                # This is O(K + M) where K is len(affected) and M is max_sum.
                # M=256. K~100. So ~400 ops.
                # 1000 steps * 400 ops = 400,000 ops.
                # This might be okay.
                
                # But we need to do this correctly.
                
                # Let's rewrite the inner loop using bincount.
                
                # Re-initialize the loop structure to use bincount.
                
                # We'll break out and re-implement the SA loop properly.
                
                break # Break inner loop to restart with better implementation
        
        return best_s, best_score

    # Re-implement SA with bincount optimization
    
    def run_sa_optimized(s_init, time_budget):
        s = s_init.copy()
        
        sums = np.zeros(num_aps, dtype=np.int32)
        for r, indices in enumerate(ap_indices):
            sums[r] = np.sum(s[indices])
            
        max_possible_sum = N
        hist = np.zeros(max_possible_sum + 1, dtype=np.int32)
        hist[np.abs(sums)] += 1
        
        current_max = int(np.max(np.abs(sums)))
        while hist[current_max] == 0:
            current_max -= 1
            
        best_s = s.copy()
        best_score = current_max
        
        rng = random.Random(123)
        
        T = 2.0
        cooling_rate = 0.9999 # Slow cooling
        
        # Run until time is up
        while time.time() - start_time < time_limit:
            # Batch of steps
            for _ in range(100):
                if time.time() - start_time > time_limit:
                    break
                
                i = rng.randint(0, N - 1)
                
                old_val = s[i]
                delta_val = -2 * old_val
                affected_aps = aps_containing_np[i]
                
                if len(affected_aps) == 0:
                    continue
                
                old_sums = sums[affected_aps].copy()
                new_sums = old_sums + delta_val
                
                # Update hist using bincount
                old_abs = np.abs(old_sums)
                new_abs = np.abs(new_sums)
                
                # Subtract old counts
                hist[old_abs] -= 1
                # Add new counts
                hist[new_abs] += 1
                
                # Update sums and s
                sums[affected_aps] = new_sums
                s[i] = -old_val
                
                # Update current_max
                # Check if max increased
                # max(new_abs)
                max_new_abs = int(np.max(new_abs))
                if max_new_abs > current_max:
                    current_max = max_new_abs
                else:
                    # Check if current_max is still valid
                    if hist[current_max] == 0:
                        # Find new max
                        # We can scan down from current_max
                        # Or use np.nonzero(hist)[0].max()
                        # Scanning down is faster if max doesn't drop much.
                        while current_max > 0 and hist[current_max] == 0:
                            current_max -= 1
                        
                # Accept/Reject
                if current_max < best_score:
                    best_score = current_max
                    best_s = s.copy()
                
                # Cooling
                T *= cooling_rate
                
                # If T is very low, we are in greedy mode.
                # If current_max == best_score, we might want to explore more?
                # Standard SA: accept if delta < 0 or rand < exp(-delta/T)
                
                # Here, "delta" is increase in max discrepancy.
                # If current_max < old_max (before flip), delta is negative -> Accept.
                # If current_max > old_max, delta is positive -> Maybe reject.
                
                # We need to track old_max.
                # We didn't store it. Let's add that.
                
                # Wait, the loop above already accepted the move.
                # We need to implement the rejection logic.
                
                # Let's re-structure to calculate delta first.
                
                # This implies I need to rewrite the loop again.
                
                break
        
        return best_s, best_score

    # Let's write a clean, correct SA implementation.
    
    def run_sa_clean(s_init, time_budget):
        s = s_init.copy()
        
        sums = np.zeros(num_aps, dtype=np.int32)
        for r, indices in enumerate(ap_indices):
            sums[r] = np.sum(s[indices])
            
        max_possible_sum = N
        hist = np.zeros(max_possible_sum + 1, dtype=np.int32)
        hist[np.abs(sums)] += 1
        
        current_max = int(np.max(np.abs(sums)))
        while hist[current_max] == 0:
            current_max -= 1
            
        best_s = s.copy()
        best_score = current_max
        
        rng = random.Random(42)
        
        T = 5.0
        # We will run for a fixed number of iterations, determined by time budget.
        # Estimate: 10k steps/sec? 
        # Let's just run until time is up.
        
        # To make cooling work, we need to know total steps.
        # Let's assume we can do ~50,000 steps/sec.
        # So in 50s, 2.5M steps.
        # T_start = 5, T_end = 0.1
        # cooling = (T_end/T_start)^(1/total_steps)
        
        # Let's just use a simple time-based cooling.
        # T = T_start * exp(-k * elapsed)
        
        while time.time() - start_time < time_limit:
            i = rng.randint(0, N - 1)
            
            old_val = s[i]
            delta_val = -2 * old_val
            affected_aps = aps_containing_np[i]
            
            if len(affected_aps) == 0:
                continue
            
            old_sums = sums[affected_aps].copy()
            new_sums = old_sums + delta_val
            
            # Calculate new max
            # We need to update hist to get new max.
            old_abs = np.abs(old_sums)
            new_abs = np.abs(new_sums)
            
            # Update hist
            hist[old_abs] -= 1
            hist[new_abs] += 1
            
            # Find new_max
            # If max increased, it's max(new_abs)
            # If max decreased, it's the highest k with hist[k]>0
            # Optimization: 
            # max_new_abs = np.max(new_abs)
            # if max_new_abs > current_max:
            #     new_max = max_new_abs
            # else:
            #     if hist[current_max] == 0:
            #         # scan down
            #         new_max = current_max
            #         while new_max > 0 and hist[new_max] == 0:
            #             new_max -= 1
            #     else:
            #         new_max = current_max
            #     # But what if max decreased but not to current_max? 
            #     # e.g. current_max=10, we changed one AP from 10 to 8. hist[10] becomes 0. 
            #     # We scan down to 8. Correct.
            
            max_new_abs = int(np.max(new_abs))
            if max_new_abs > current_max:
                new_max = max_new_abs
            else:
                if hist[current_max] == 0:
                    new_max = current_max
                    while new_max > 0 and hist[new_max] == 0:
                        new_max -= 1
                else:
                    new_max = current_max
            
            # Delta
            delta = new_max - current_max
            
            # Acceptance
            accept = False
            if delta <= 0:
                accept = True
            else:
                # T based on elapsed time
                elapsed = time.time() - start_time
                # Simple cooling: T = 5 * exp(-elapsed/10)
                T = 5.0 * math.exp(-elapsed / 10.0)
                if T < 0.01:
                    T = 0.01
                prob = math.exp(-delta / T)
                accept = rng.random() < prob
            
            if accept:
                sums[affected_aps] = new_sums
                s[i] = -old_val
                current_max = new_max
                
                if current_max < best_score:
                    best_score = current_max
                    best_s = s.copy()
            else:
                # Revert hist
                hist[old_abs] += 1
                hist[new_abs] -= 1
                # sums and s are not changed
            
        return best_s, best_score

    # Main Loop
    best_global_s = None
    best_global_score = float('inf')
    
    rng_global = random.Random(42)
    
    s_thue = thue_morse(N)
    s_random1 = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
    
    inits = [s_thue, s_random1]
    
    # Run SA for each init
    # We will split the time budget between restarts.
    # Let's try 3 restarts.
    
    time_per_run = (time_limit - 1.0) / 3.0
    
    for s_init in inits:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 2.0:
            break
            
        run_budget = min(time_per_run, remaining)
        
        s_best_run, score_best_run = run_sa_clean(s_init, run_budget)
        
        if score_best_run < best_global_score:
            best_global_score = score_best_run
            best_global_s = s_best_run
            
        if best_global_score <= 1:
            break

    if best_global_s is None:
        best_global_s = thue_morse(N)
        
    return [int(x) for x in best_global_s]
