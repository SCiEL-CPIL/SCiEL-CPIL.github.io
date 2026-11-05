import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = 128
    target_min = 128
    target_max = 256
    target_sums = range(target_min, target_max + 1)
    
    best_peak = float('inf')
    best_A = None
    
    # Helper to calculate counts
    def get_counts(A_set):
        counts = np.zeros(257, dtype=np.int32)
        A_list = list(A_set)
        for i in range(len(A_list)):
            a = A_list[i]
            for j in range(i, len(A_list)):
                b = A_list[j]
                s = a + b
                if target_min <= s <= target_max:
                    counts[s] += 1
        return counts

    def evaluate(A_set):
        counts = get_counts(A_set)
        missing = 0
        for s in target_sums:
            if counts[s] == 0:
                missing += 1
        peak = np.max(counts[target_min:target_max+1])
        # Objective: minimize missing, then peak, then sum of squares
        # Return tuple for comparison
        return (missing, peak, np.sum(counts[target_min:target_max+1]**2))

    # Initial random generation
    def random_init(k):
        A = set(random.sample(range(n+1), k))
        return A

    # Repair function: ensure all sums are covered
    def repair(A_set):
        A_list = list(A_set)
        counts = get_counts(A_set)
        
        while True:
            missing_sums = [s for s in target_sums if counts[s] == 0]
            if not missing_sums:
                break
            
            # Find best element to add
            best_x = None
            best_gain = -1
            best_new_counts = None
            
            # Try adding each x not in A
            # Optimization: only try x that can form a sum with existing A
            candidates = [x for x in range(n+1) if x not in A_set]
            
            for x in candidates:
                new_counts = counts.copy()
                # Add pairs (x, a) for a in A
                for a in A_list:
                    s = x + a
                    if target_min <= s <= target_max:
                        new_counts[s] += 1
                # Also pair (x, x)
                if 2*x in target_sums:
                    new_counts[2*x] += 1
                
                # Check if this covers any missing sums
                gain = 0
                for s in missing_sums:
                    if new_counts[s] > 0:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    best_new_counts = new_counts
            
            if best_x is None or best_gain == 0:
                # Cannot cover with single addition? 
                # This shouldn't happen if we have enough elements.
                # If stuck, add random element?
                # Actually, if A is small, we might need to add multiple.
                # Let's just add a random element from candidates if stuck.
                if candidates:
                    best_x = random.choice(candidates)
                    # Recalculate counts properly
                    A_set.add(best_x)
                    A_list = list(A_set)
                    counts = get_counts(A_set)
                    continue
                else:
                    break
            
            A_set.add(best_x)
            A_list = list(A_set)
            counts = best_new_counts
        
        return A_set

    # Local Search
    def local_search(A_set, time_limit):
        A_set = set(A_set)
        A_list = list(A_set)
        counts = get_counts(A_set)
        current_obj = evaluate(A_set)
        
        while time.time() < time_limit:
            improved = False
            # Try swaps
            # Shuffle candidates to avoid bias
            candidates = [x for x in range(n+1) if x not in A_set]
            random.shuffle(candidates)
            
            for a_out in A_list:
                if time.time() > time_limit:
                    break
                
                # Remove a_out
                A_set.remove(a_out)
                A_list.remove(a_out)
                new_counts = get_counts(A_set)
                
                for a_in in candidates:
                    # Add a_in
                    A_set.add(a_in)
                    A_list.append(a_in)
                    
                    # Update counts efficiently? 
                    # For now, recalculate or update incrementally.
                    # Incremental:
                    # We removed a_out. We need to decrement counts for pairs involving a_out.
                    # We added a_in. We need to increment counts for pairs involving a_in.
                    # This is complex to do right with numpy slices easily in a tight loop without bugs.
                    # Let's just use get_counts for simplicity first, if too slow, optimize.
                    # get_counts is O(K^2). K~30. 900 ops.
                    # 30 * 100 swaps = 3000 * 900 = 2.7M ops. ~0.3s per iteration.
                    # 60s / 0.3s = 200 iterations. Acceptable.
                    
                    new_counts_full = get_counts(A_set)
                    new_obj = (
                        sum(1 for s in target_sums if new_counts_full[s] == 0),
                        np.max(new_counts_full[target_min:target_max+1]),
                        np.sum(new_counts_full[target_min:target_max+1]**2)
                    )
                    
                    if new_obj < current_obj:
                        current_obj = new_obj
                        improved = True
                        # Keep the swap
                        # A_set and A_list are already updated
                        counts = new_counts_full
                        break
                    else:
                        # Revert
                        A_set.remove(a_in)
                        A_list.pop()
                
                # Revert a_out removal if we didn't swap in something
                if a_out not in A_set:
                    A_set.add(a_out)
                    A_list.append(a_out)
                    # Recalculate counts if we modified A_set in the loop but reverted?
                    # If we broke out of inner loop with improvement, a_in is in A_set.
                    # If we didn't improve, a_out is back in A_set.
                    # We need counts to be consistent.
                    if improved:
                         # We accepted a swap. a_out is out, a_in is in.
                         # counts is updated.
                         pass
                    else:
                         # We reverted. a_out is in.
                         counts = get_counts(A_set)

            if not improved:
                # Try random restart or perturbation?
                # If stuck, break to try a new random init
                break
        
        return A_set, current_obj

    # Main loop
    for i in range(100):
        if time.time() > deadline:
            break
        
        # Try different sizes
        k = random.randint(25, 45)
        A_init = random_init(k)
        
        # Repair
        A_repaired = repair(A_init)
        
        # Evaluate
        obj = evaluate(A_repaired)
        if obj[0] > 0:
            continue # Not valid
        
        peak = obj[1]
        if peak < best_peak:
            best_peak = peak
            best_A = set(A_repaired)
        
        # Local search
        time_left = deadline - time.time()
        if time_left > 1:
            A_opt, obj_opt = local_search(A_repaired, time_left - 1)
            if obj_opt[0] == 0 and obj_opt[1] < best_peak:
                best_peak = obj_opt[1]
                best_A = set(A_opt)
        
        if best_peak <= 4:
             # Hard to beat 4? Maybe 3 is possible?
             # If we reach 4, we can try to squeeze for 3.
             # But 4 is already much better than 33.
             pass

    if best_A is None:
        # Fallback to seed
        return list(range(64, 129))
    
    return list(best_A)
