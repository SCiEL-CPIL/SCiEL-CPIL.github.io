import time
import random
import math

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0  # Safety margin
    
    # Precompute APs
    # AP is defined by start (0-indexed) and step d.
    # Indices: start, start+d, ..., < n
    aps = []
    index_to_aps = [[] for _ in range(n)]
    
    for d in range(1, n + 1):
        for start in range(0, n, d):
            idxs = list(range(start, n, d))
            if idxs:
                ap_id = len(aps)
                aps.append(idxs)
                for idx in idxs:
                    index_to_aps[idx].append(ap_id)
    
    num_aps = len(aps)
    
    # Initialize signs randomly
    rng = random.Random(42)
    signs = [rng.choice([-1, 1]) for _ in range(n)]
    
    # Initialize sums
    ap_sums = [0] * num_aps
    for k, idxs in enumerate(aps):
        s = 0
        for i in idxs:
            s += signs[i]
        ap_sums[k] = s
        
    # Initial max
    current_max = max(abs(s) for s in ap_sums)
    
    # Track count of APs achieving current_max to avoid frequent rescans
    max_count = sum(1 for s in ap_sums if abs(s) == current_max)
    
    best_signs = signs[:]
    best_max = current_max
    
    # Simulated Annealing
    T = 5.0
    cooling_rate = 0.999
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Pick random index to flip
        i = rng.randrange(n)
        old_sign = signs[i]
        new_sign = -old_sign
        
        # Calculate change in sums
        # delta = -2 * old_sign
        delta = -2 * old_sign
        
        # Tentatively update sums and track max change
        old_max = current_max
        new_max = current_max
        max_count_new = max_count
        
        affected_aps = index_to_aps[i]
        
        # We need to revert if rejected, so store old values?
        # Or just update and revert.
        
        # Optimization: Check if new_max increases
        # If any new abs sum > old_max, new_max is that value.
        
        changed_max_count = 0
        # Store old values of affected APs to revert if needed
        # Since we only change 256 APs, storing them is cheap.
        old_vals = {}
        
        for k in affected_aps:
            old_val = ap_sums[k]
            old_vals[k] = old_val
            new_val = old_val + delta
            ap_sums[k] = new_val
            
            abs_old = abs(old_val)
            abs_new = abs(new_val)
            
            # Update max count logic
            if abs_old == old_max:
                max_count_new -= 1
            if abs_new == old_max:
                max_count_new += 1
                
            # Check if new max is higher
            if abs_new > new_max:
                new_max = abs_new
        
        # If new_max > old_max, we definitely accept (or SA might reject, but usually we accept improvements)
        # If new_max < old_max, we need to check if old_max is still present.
        # If max_count_new == 0 and new_max < old_max, we must rescan to find true new_max.
        
        actual_new_max = new_max
        if max_count_new == 0 and actual_new_max < old_max:
            # Rescan all APs to find new max
            actual_new_max = max(abs(s) for s in ap_sums)
            # Recalculate max_count for the new max
            max_count_new = sum(1 for s in ap_sums if abs(s) == actual_new_max)
        elif actual_new_max > old_max:
            # New max found, update count for this new max
            # Note: new_max was set to the max of the affected APs.
            # It's possible an unaffected AP had the same value? No, if new_max > old_max, it's strictly higher.
            # So max_count_new should be the count of affected APs with abs(val) == new_max.
            # But wait, if new_max > old_max, then no other AP (affected or not) could have had that value before.
            # And among affected APs, we counted how many have abs(val) == old_max.
            # We need count of abs(val) == new_max.
            max_count_new = sum(1 for k in affected_aps if abs(ap_sums[k]) == new_max)
            # Unaffected APs cannot have abs(val) == new_max because new_max > old_max >= all old values.
        
        # Acceptance criteria
        cost_diff = actual_new_max - old_max
        
        if cost_diff < 0:
            accept = True
        else:
            if T < 1e-5:
                accept = False
            else:
                # Avoid overflow
                if cost_diff / T > 50:
                    prob = 0.0
                else:
                    prob = math.exp(-cost_diff / T)
                accept = rng.random() < prob
        
        if accept:
            signs[i] = new_sign
            current_max = actual_new_max
            max_count = max_count_new
            
            if current_max < best_max:
                best_max = current_max
                best_signs = signs[:]
        else:
            # Revert
            for k, val in old_vals.items():
                ap_sums[k] = val
            # signs[i] remains old_sign
            # current_max, max_count remain old values
        
        T *= cooling_rate

    return best_signs
