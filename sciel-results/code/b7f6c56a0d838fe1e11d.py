import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    n = 256

    # 1. Pre-compute APs
    # APs are defined by start r, step d.
    # Indices: r, r+d, r+2d, ... < n
    # We only care about APs with length >= 2? 
    # "length including one". Singletons have sum +/-1. 
    # If we can make all APs length >= 2 have sum 0 or 1, and singletons are 1, 
    # then max is 1. 
    # But singletons are always +/-1. So min possible max is 1.
    # But as established, 1 is impossible for n=256 due to even steps.
    # So we aim for 2 or 3.
    
    aps = []
    ap_membership = [[] for _ in range(n)]
    
    # Store AP indices and their current sums
    # We need to iterate d from 1 to n//2
    for d in range(1, n):
        for r in range(d):
            if r >= n:
                break
            # Generate indices
            indices = list(range(r, n, d))
            if len(indices) >= 2: # Task says "including one", but singleton is trivially 1.
                # We should include singletons to be safe? 
                # "any start, positive step and length including one"
                # If we ignore singletons, we might miss the constraint if max is 1.
                # But max is definitely > 1. So singletons (sum 1) won't be the bottleneck 
                # unless we achieve 0 for all others, which is impossible.
                # So we can ignore singletons for optimization, 
                # but the final check must include them? 
                # The evaluator checks ALL. 
                # If my optimized sequence has max AP sum 2 (for len>=2), 
                # and singletons are 1, the overall max is 2.
                # If I achieve max AP sum 1 (for len>=2), overall max is 1.
                # So I should include singletons in the "APs" list if I want to 
                # potentially reach 1. 
                # However, singletons don't change. 
                # So I can just add 1 to the final result if I don't track them.
                # Let's track only len >= 2 to save memory/time, 
                # and remember that singletons contribute 1.
                pass
            
            # Actually, let's just include all APs with len >= 1.
            # It's only 8000 APs.
            ap_idx = len(aps)
            aps.append(indices)
            for idx in indices:
                ap_membership[idx].append(ap_idx)

    num_aps = len(aps)
    
    # Initialize signs
    # Use a random start, or better, a Thue-Morse variant
    # Thue-Morse: x_i = (-1)^(popcount(i))
    # This has good low-frequency properties.
    signs = [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    
    # Initialize AP sums
    ap_sums = np.zeros(num_aps, dtype=np.int32)
    for i, ap in enumerate(aps):
        s = 0
        for idx in ap:
            s += signs[idx]
        ap_sums[i] = s
            
    # Track max absolute sum
    # We need to update this efficiently.
    # Let's maintain a count of absolute sums.
    # Max possible sum is n.
    max_possible_sum = n
    abs_sum_counts = np.zeros(max_possible_sum + 1, dtype=np.int32)
    
    for s in ap_sums:
        abs_sum_counts[abs(s)] += 1
        
    current_max = 0
    for k in range(max_possible_sum, -1, -1):
        if abs_sum_counts[k] > 0:
            current_max = k
            break
            
    # Note: Singletons are length 1. Their sum is always 1.
    # So the true max discrepancy is max(current_max, 1).
    # Since current_max will likely be > 1, we can just track current_max.
    # But if we get current_max = 0 or 1, we must return 1.
    
    best_signs = signs[:]
    best_disc = max(current_max, 1)
    
    # Helper to update state after a flip
    def flip(idx):
        nonlocal current_max
        old_val = signs[idx]
        new_val = -old_val
        signs[idx] = new_val
        
        # Update APs containing idx
        for ap_idx in ap_membership[idx]:
            old_sum = ap_sums[ap_idx]
            old_abs = abs(old_sum)
            
            # Update count
            abs_sum_counts[old_abs] -= 1
            
            # New sum
            new_sum = old_sum + 2 * new_val # old_sum - 2*old_val + 2*new_val = old_sum + 2(new_val - old_val)
            # Since new_val = -old_val, diff is 2*new_val - 2*old_val = -4*old_val? 
            # No. 
            # Sum changes by (new_val - old_val) = 2*new_val.
            # If old was 1, new is -1. Sum changes by -2.
            # If old was -1, new is 1. Sum changes by +2.
            # So new_sum = old_sum + 2 * new_val.
            
            ap_sums[ap_idx] = new_sum
            new_abs = abs(new_sum)
            
            abs_sum_counts[new_abs] += 1
            
            # Update current_max
            if new_abs > current_max:
                current_max = new_abs
            elif old_abs == current_max and abs_sum_counts[old_abs] == 0:
                # Need to find new max
                # This is rare, so linear scan down is okay
                while current_max > 0 and abs_sum_counts[current_max] == 0:
                    current_max -= 1
                    
    # Simulated Annealing
    rng = random.Random(42)
    temp = 5.0
    cool = 0.998
    no_improve = 0
    max_no_improve = 1000
    
    # Initial local search
    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        # Try flipping each bit
        for i in range(n):
            old_max = current_max
            flip(i)
            if current_max < old_max:
                improved = True
                if current_max < best_disc:
                    best_disc = current_max
                    best_signs = signs[:]
            else:
                flip(i) # Undo
                
    # SA Loop
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        # Choose a move
        # Bias towards flipping bits in "bad" APs?
        # Simple random first.
        idx = rng.randint(0, n - 1)
        
        old_max = current_max
        flip(idx)
        
        delta = current_max - old_max
        
        if delta < 0:
            # Improvement
            no_improve = 0
            if current_max < best_disc:
                best_disc = current_max
                best_signs = signs[:]
        else:
            # No improvement or worse
            if delta == 0:
                # Accept to move around plateau
                no_improve += 1
            else:
                # Worse
                if temp > 0.1:
                    prob = math.exp(-delta / temp)
                    if rng.random() < prob:
                        no_improve += 1
                    else:
                        flip(idx) # Undo
                        no_improve += 1
                else:
                    flip(idx) # Undo
                    no_improve += 1
                    
        temp *= cool
        
    # Final Polish: Hill Climbing
    signs = best_signs[:]
    # Re-init state for best_signs
    ap_sums = np.zeros(num_aps, dtype=np.int32)
    for i, ap in enumerate(aps):
        s = 0
        for idx in ap:
            s += signs[idx]
        ap_sums[i] = s
            
    abs_sum_counts = np.zeros(max_possible_sum + 1, dtype=np.int32)
    for s in ap_sums:
        abs_sum_counts[abs(s)] += 1
        
    current_max = 0
    for k in range(max_possible_sum, -1, -1):
        if abs_sum_counts[k] > 0:
            current_max = k
            break
            
    # Local search again
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for i in range(n):
            old_max = current_max
            flip(i)
            if current_max < old_max:
                improved = True
                if current_max < best_disc:
                    best_disc = current_max
                    best_signs = signs[:]
            else:
                flip(i) # Undo

    # Ensure we return the best found
    return best_signs
