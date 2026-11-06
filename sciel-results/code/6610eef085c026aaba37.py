import numpy as np
import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 88.0  # Safety margin for 90s limit
    
    # 1. Identify Critical Arithmetic Progressions
    # The maximum sum over ALL APs is dominated by APs with large step sizes (sparse) 
    # or long lengths (dense). 
    # Short APs with small steps (e.g., consecutive integers) tend to have lower 
    # maximum absolute sums in optimal configurations compared to the "tail" of 
    # large-step APs which behave more like independent random variables or 
    # structured subsets that can accumulate larger sums if not carefully balanced.
    # 
    # Strategy: Select a subset of "critical" APs that are most likely to 
    # determine the global maximum.
    # - All APs with step s >= 16 (these are sparse and often have higher variance)
    # - All APs with length L >= 16 (these are long and dense)
    # - Additionally, include a random sample of "medium" APs to ensure coverage.
    
    critical_progs = []
    seen = set()
    
    # We represent a progression by its (start, step) tuple. 
    # The indices are determined by start, step, and n.
    for a in range(n):
        for s in range(1, n):
            # Calculate length
            # indices: a, a+s, ..., a+(k-1)s < n
            # k = floor((n-1-a)/s) + 1
            k = (n - 1 - a) // s + 1
            if k < 1:
                continue
            
            # Heuristic for "criticality":
            # 1. Large step: s >= 16
            # 2. Long length: k >= 16
            # 3. Or, if we haven't added enough, add some random ones later.
            
            is_critical = (s >= 16 or k >= 16)
            
            if is_critical:
                key = (a, s)
                if key not in seen:
                    seen.add(key)
                    critical_progs.append((a, s))
    
    # To ensure we don't miss any global max that might come from a "medium" AP,
    # we will also evaluate a fixed set of random APs in the final check, 
    # but the local search will only optimize against the critical set.
    # However, if the critical set is too small, we might optimize locally but 
    # fail globally. 
    # Let's add a random sample of other APs to the critical set to balance.
    rng = np.random.RandomState(42)
    
    # Estimate total number of APs
    # Sum_{a=0}^{n-1} Sum_{s=1}^{n-a-1} 1 approx n^2/2 = 32768
    # Critical set size is likely a few thousand.
    
    # Let's add 500 random APs to the critical set to ensure robustness
    all_ap_keys = []
    for a in range(n):
        for s in range(1, n):
            k = (n - 1 - a) // s + 1
            if k >= 1:
                all_ap_keys.append((a, s))
                
    # Shuffle and pick some that are not already critical
    rng.shuffle(all_ap_keys)
    for a, s in all_ap_keys:
        if (a, s) not in seen:
            seen.add((a, s))
            critical_progs.append((a, s))
            if len(critical_progs) > len(seen) - 1 + 500: # Just add up to 500 more
                break
    
    Np = len(critical_progs)
    
    # 2. Build Data Structures for Fast Updates
    # For each position i (0 to n-1), store the list of (prog_idx, sign_in_prog) 
    # where sign_in_prog is effectively 1 (since M[i, p] is 1).
    # When we flip sign[i], for each prog p containing i:
    #   sums[p] += -2 * old_sign[i]  (if we flip +1 to -1, sum decreases by 2)
    #   sums[p] +=  2 * old_sign[i]  (if we flip -1 to +1, sum increases by 2)
    #   Actually: new_sum = old_sum - 2 * old_sign[i]
    
    pos_to_progs = [[] for _ in range(n)]
    for pid, (a, s) in enumerate(critical_progs):
        for i in range(a, n, s):
            pos_to_progs[i].append(pid)
    
    # Precompute the indices for each critical progression for final evaluation
    # We don't need the full matrix M if we use incremental updates.
    # But we need to compute initial sums.
    
    # 3. Optimization Loop
    best_signs = None
    best_score = float('inf')
    
    # Multi-restart
    for restart in range(100):
        if time.time() > deadline - 2.0:
            break
            
        # Initialize
        signs = rng.choice([-1, 1], size=n, dtype=np.int8)
        
        # Compute initial sums for critical progs
        sums = np.zeros(Np, dtype=np.int32)
        for pid, (a, s) in enumerate(critical_progs):
            # Sum signs at a, a+s, ...
            # Vectorized sum
            idx = np.arange(a, n, s)
            sums[pid] = np.sum(signs[idx])
            
        current_score = int(np.max(np.abs(sums)))
        
        # Check global max against ALL APs? 
        # Doing this every step is too slow. 
        # We will do a "Global Check" every 1000 local steps or if we hit a new local min.
        # For now, track local score.
        
        if current_score < best_score:
            # Verify against all APs? 
            # To save time, we assume critical set is a good proxy. 
            # We will do a full verification at the end or periodically.
            best_score = current_score
            best_signs = signs.copy()
        
        # Local Search: Histogram-based / Incremental
        # We want to find flips that reduce the max abs sum.
        # Instead of checking all 256 positions, we can focus on positions 
        # that are in the current worst progressions.
        
        improved = True
        steps_in_restart = 0
        
        while improved and time.time() < deadline - 2.0:
            improved = False
            steps_in_restart += 1
            
            if current_score <= 1:
                break
                
            # Find worst progression in critical set
            abs_sums = np.abs(sums)
            p_worst = np.argmax(abs_sums)
            worst_val = abs_sums[p_worst]
            
            # Get positions in this progression
            a, s = critical_progs[p_worst]
            positions = np.arange(a, n, s)
            
            # Try flipping each position in this progression
            # We want to minimize the new max abs sum.
            # Approximation: Minimize the new abs sum of p_worst, 
            # while checking if it increases any other "very bad" progression.
            
            best_i = -1
            best_new_local_score = current_score
            
            # To be fast, we only evaluate a few candidates or all if short.
            # If the progression is long, we can't evaluate all flips' global impact easily.
            # But we can evaluate the impact on p_worst and its neighbors.
            
            # Let's try the flip that most reduces |sum[p_worst]|
            # sum_p = sums[p_worst]
            # If we flip i, new_sum_p = sum_p - 2 * signs[i]
            # We want new_sum_p to be close to 0.
            
            current_sum_p = sums[p_worst]
            
            # Calculate new sum for each i in positions
            # signs_pos = signs[positions]
            # new_sums_p = current_sum_p - 2 * signs_pos
            # diff = abs(new_sums_p)
            
            signs_pos = signs[positions]
            new_sums_p = current_sum_p - 2 * signs_pos
            abs_new_sums_p = np.abs(new_sums_p)
            
            # Pick the i that minimizes abs_new_sums_p
            min_idx_in_pos = np.argmin(abs_new_sums_p)
            i_to_flip = positions[min_idx_in_pos]
            
            # Evaluate this flip's impact on the critical set
            old_sign = signs[i_to_flip]
            # delta for sums: for each prog p containing i, sums[p] -= 2 * old_sign
            # Since we are flipping, the change is -2 * old_sign
            delta_val = -2 * old_sign
            
            # We need to check if this flip increases the global max.
            # Instead of updating all sums and checking max (O(Np)), 
            # we can check the specific progressions that might become worse.
            # However, Np is ~few thousand. Updating sums and checking max is O(Np).
            # 256 * 1000 steps * 5000 ops = 1.2e9 ops. Might be too slow in Python.
            
            # Optimization: Only update the sums for the progressions containing i.
            # But we still need to know the new MAX.
            # We can maintain a histogram of abs(sums).
            # Histogram update: 
            #   For each prog p in pos_to_progs[i]:
            #       old_val = abs(sums[p])
            #       hist[old_val] -= 1
            #       sums[p] += delta_val
            #       new_val = abs(sums[p])
            #       hist[new_val] += 1
            #   Then find the max index in hist.
            
            # Let's implement Histogram-based tracking for speed.
            # Max possible sum is n=256. So hist size is 257.
            
            if not hasattr(construct, 'hist_initialized'):
                # This is inside the function, so we need to manage state carefully or re-init per restart
                pass
            
            # Re-initialize histogram for this restart
            hist = np.zeros(257, dtype=np.int32)
            for v in np.abs(sums):
                hist[v] += 1
            
            # Find current max from hist
            # current_score should match np.max(np.nonzero(hist)[0])
            
            # Try the flip
            # 1. Update hist and sums for progs containing i_to_flip
            pids_containing_i = pos_to_progs[i_to_flip]
            
            # Save old values for rollback if rejected
            old_sums_vals = []
            old_hist_vals = []
            
            for pid in pids_containing_i:
                old_s = sums[pid]
                old_abs = abs(old_s)
                hist[old_abs] -= 1
                new_s = old_s + delta_val
                sums[pid] = new_s
                new_abs = abs(new_s)
                hist[new_abs] += 1
                old_sums_vals.append((pid, old_s, old_abs, new_abs))
                
            # Find new max
            # Scan hist from top down
            new_score = 0
            for k in range(256, -1, -1):
                if hist[k] > 0:
                    new_score = k
                    break
            
            if new_score < current_score:
                # Accept
                signs[i_to_flip] = -old_sign
                current_score = new_score
                improved = True
                
                if current_score < best_score:
                    # We should verify against ALL APs here to be safe?
                    # It's expensive. Let's do it every 500 improvements or at end.
                    # For now, trust the critical set.
                    best_score = current_score
                    best_signs = signs.copy()
            else:
                # Reject: Rollback
                for pid, old_s, old_abs, new_abs in old_sums_vals:
                    hist[new_abs] -= 1
                    hist[old_abs] += 1
                    sums[pid] = old_s
                
                # If we can't improve by flipping the best candidate in the worst prog,
                # try a random flip to escape local minimum
                if rng.random() < 0.1: # 10% chance to perturb
                    i_rand = rng.randint(n)
                    old_sign_rand = signs[i_rand]
                    delta_rand = -2 * old_sign_rand
                    pids_rand = pos_to_progs[i_rand]
                    
                    # Update hist
                    rollback_rand = []
                    for pid in pids_rand:
                        old_s = sums[pid]
                        old_abs = abs(old_s)
                        hist[old_abs] -= 1
                        new_s = old_s + delta_rand
                        sums[pid] = new_s
                        new_abs = abs(new_s)
                        hist[new_abs] += 1
                        rollback_rand.append((pid, old_s, old_abs, new_abs))
                        
                    new_score_rand = 0
                    for k in range(256, -1, -1):
                        if hist[k] > 0:
                            new_score_rand = k
                            break
                    
                    if new_score_rand < current_score:
                        signs[i_rand] = -old_sign_rand
                        current_score = new_score_rand
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_signs = signs.copy()
                    else:
                        # Rollback
                        for pid, old_s, old_abs, new_abs in rollback_rand:
                            hist[new_abs] -= 1
                            hist[old_abs] += 1
                            sums[pid] = old_s

    # Final Polish: Verify best_signs against ALL APs and try to improve
    if best_signs is not None and time.time() < deadline - 1.0:
        signs = best_signs.copy()
        
        # Compute true max for all APs
        def get_true_max(signs):
            max_sum = 0
            for a in range(n):
                for s in range(1, n):
                    # Sum
                    # Using slicing for speed
                    idx = np.arange(a, n, s)
                    val = np.abs(np.sum(signs[idx]))
                    if val > max_sum:
                        max_sum = val
            return int(max_sum)
        
        true_score = get_true_max(signs)
        
        # If true_score > best_score, we have a gap. 
        # We can try to reduce it by targeting the specific APs that caused the high true_score.
        # Find the worst APs
        worst_aps = []
        for a in range(n):
            for s in range(1, n):
                idx = np.arange(a, n, s)
                val = np.sum(signs[idx])
                if abs(val) > best_score:
                    worst_aps.append((a, s, val))
        
        # Add these to critical set and do a few more rounds of local search
        if worst_aps:
            # Extend critical_progs with worst_aps
            for a, s, _ in worst_aps:
                if (a, s) not in seen:
                    seen.add((a, s))
                    critical_progs.append((a, s))
            Np_new = len(critical_progs)
            
            # Rebuild pos_to_progs for the new set
            pos_to_progs_new = [[] for _ in range(n)]
            for pid in range(len(critical_progs)):
                a, s = critical_progs[pid]
                for i in range(a, n, s):
                    pos_to_progs_new[i].append(pid)
            
            # Re-init sums and hist
            sums_new = np.zeros(Np_new, dtype=np.int32)
            for pid, (a, s) in enumerate(critical_progs):
                idx = np.arange(a, n, s)
                sums_new[pid] = np.sum(signs[idx])
                
            hist_new = np.zeros(257, dtype=np.int32)
            for v in np.abs(sums_new):
                hist_new[v] += 1
                
            cur_score = int(np.max(np.abs(sums_new)))
            
            # Local search on this expanded set
            for _ in range(1000):
                if time.time() > deadline - 0.5:
                    break
                if cur_score <= 1:
                    break
                
                p_worst = np.argmax(np.abs(sums_new))
                a, s = critical_progs[p_worst]
                positions = np.arange(a, n, s)
                
                current_sum_p = sums_new[p_worst]
                signs_pos = signs[positions]
                new_sums_p = current_sum_p - 2 * signs_pos
                abs_new_sums_p = np.abs(new_sums_p)
                min_idx = np.argmin(abs_new_sums_p)
                i_to_flip = positions[min_idx]
                
                old_sign = signs[i_to_flip]
                delta_val = -2 * old_sign
                pids = pos_to_progs_new[i_to_flip]
                
                rollback = []
                for pid in pids:
                    old_s = sums_new[pid]
                    old_abs = abs(old_s)
                    hist_new[old_abs] -= 1
                    new_s = old_s + delta_val
                    sums_new[pid] = new_s
                    new_abs = abs(new_s)
                    hist_new[new_abs] += 1
                    rollback.append((pid, old_s, old_abs, new_abs))
                    
                new_score = 0
                for k in range(256, -1, -1):
                    if hist_new[k] > 0:
                        new_score = k
                        break
                        
                if new_score < cur_score:
                    signs[i_to_flip] = -old_sign
                    cur_score = new_score
                    if cur_score < best_score:
                        best_score = cur_score
                        best_signs = signs.copy()
                else:
                    for pid, old_s, old_abs, new_abs in rollback:
                        hist_new[new_abs] -= 1
                        hist_new[old_abs] += 1
                        sums_new[pid] = old_s

    return [int(x) for x in best_signs]
