import time
import random
import math

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin

    # 1. Precompute APs
    # AP is (start, step)
    # We store the list of indices for each AP to compute initial sums easily
    # And we store for each index i, the list of AP indices that contain i
    
    ap_indices = []
    ap_start_step = []
    
    # Generate all APs of length >= 2
    # Note: Singletons are length 1. Sum is always 1.
    # We can ignore singletons in the optimization if we know max >= 1.
    # But we must ensure we don't go below 1.
    # Actually, if we optimize for length >= 2, and the max is 2, but a singleton is 1, the global max is 2.
    # If we optimize for length >= 2 and get max 1, the global max is 1.
    # So we can just ignore singletons in the histogram, as they don't change.
    
    for d in range(1, N + 1):
        for start in range(0, N - d + 1):
            terms = list(range(start, N, d))
            if len(terms) > 1:
                ap_indices.append(terms)
                ap_start_step.append((start, d))
                
    num_aps = len(ap_indices)
    
    # Map index -> list of AP indices containing it
    aps_containing = [[] for _ in range(N)]
    for j, terms in enumerate(ap_indices):
        for i in terms:
            aps_containing[i].append(j)

    # Helper to compute initial sums and histogram
    def compute_state(signs):
        # signs is list of 1/-1
        ap_sums = [0] * num_aps
        for j, terms in enumerate(ap_indices):
            s = 0
            for i in terms:
                s += signs[i]
            ap_sums[j] = s
        
        # Histogram of absolute sums
        # Max possible sum is N
        hist = [0] * (N + 1)
        max_val = 0
        for s in ap_sums:
            abs_s = abs(s)
            hist[abs_s] += 1
            if abs_s > max_val:
                max_val = abs_s
        return ap_sums, hist, max_val

    def try_flip(signs, i, ap_sums, hist, current_max):
        """
        Tries to flip sign at index i.
        Returns (accepted, new_signs, new_ap_sums, new_hist, new_max)
        If accepted, signs is modified in place.
        """
        old_sign = signs[i]
        new_sign = -old_sign
        delta = 2 * old_sign # sum changes by -2 * old_sign? 
        # sum_new = sum_old + (new_sign - old_sign) * 1
        # new_sign - old_sign = -2 * old_sign
        # So delta = -2 * old_sign
        
        # Calculate new max
        # We need to update hist for all APs containing i
        # We can simulate the changes first to see if it's better
        
        # To avoid modifying state if rejected, we can calculate the new max
        # without modifying the global hist/ap_sums, or we modify and revert.
        # Modify and revert is faster if we use local variables for hist changes?
        # Hist is a list. We can track changes.
        
        # Changes:
        # For each ap_idx in aps_containing[i]:
        #   old_sum = ap_sums[ap_idx]
        #   new_sum = old_sum + delta
        #   old_abs = abs(old_sum)
        #   new_abs = abs(new_sum)
        #   hist[old_abs] -= 1
        #   hist[new_abs] += 1
        
        # We need to know the new max.
        # The new max is either:
        # 1. The old max, if hist[old_max] > 0 after removals
        # 2. A value lower than old max, if hist[old_max] becomes 0
        # 3. A value higher than old max, if any new_abs > old_max
        
        # Let's track the new max efficiently.
        # max_candidate = 0
        # We don't want to scan the whole hist.
        
        # Optimization:
        # The new max can only be:
        # - current_max (if it still exists)
        # - max of new_abs values for the changed APs
        # - next highest value in hist if current_max disappears
        
        # Let's just update a copy of the relevant hist parts?
        # Or, since N is small, we can just do a full scan of hist if current_max disappears?
        # Or we can maintain a set of non-zero hist values?
        
        # Simpler:
        # 1. Determine if current_max will disappear.
        #    Count how many APs with abs_sum == current_max are being modified.
        #    If count == hist[current_max], then current_max disappears.
        
        # Let's just calculate the new max by checking:
        # - max(new_abs_values)
        # - current_max (if hist[current_max] > 0 after updates)
        # - If current_max disappears, we need the next highest non-zero in hist.
        
        # To make this fast, we can pre-compute the "next highest" logic?
        # Or just iterate down from current_max - 1.
        
        # Let's try a different approach:
        # Since N=256, and we do many steps, let's just do the update and if we reject, revert.
        # But we need to know if we accept BEFORE updating?
        # No, we can update, check max, and if worse, revert.
        
        # Reverting requires storing the old values.
        
        old_sums_changed = []
        old_hist_values = {} # abs_val -> old_count
        
        new_max = current_max
        
        # 1. Check if current_max disappears
        # Count how many of the changing APs have abs_sum == current_max
        count_max_old = 0
        for ap_idx in aps_containing[i]:
            if abs(ap_sums[ap_idx]) == current_max:
                count_max_old += 1
                
        hist_current_max_new = hist[current_max] - count_max_old
        
        # 2. Calculate new_abs for all changing APs
        new_abs_vals = []
        for ap_idx in aps_containing[i]:
            old_sum = ap_sums[ap_idx]
            new_sum = old_sum + delta
            new_abs = abs(new_sum)
            new_abs_vals.append(new_abs)
            
        max_new_abs = max(new_abs_vals) if new_abs_vals else 0
        
        # Determine new_max
        if hist_current_max_new > 0:
            new_max = current_max
        else:
            # Current max is gone.
            # New max is max(max_new_abs, next_highest_in_hist)
            # next_highest_in_hist: we need to find the largest k < current_max with hist[k] > 0
            # We can just scan down from current_max - 1
            # But we must account for the fact that hist values for keys in new_abs_vals are increasing.
            
            # Actually, simpler:
            # The new max is the max of:
            # - max_new_abs
            # - any k < current_max such that hist[k] > 0 (after updates)
            
            # Let's just calculate the exact new max by scanning the hist?
            # No, that's O(N).
            
            # Let's assume the new max is either max_new_abs or the previous second highest.
            # To be safe and simple, let's just update the hist and find the max.
            # Since we are in a tight loop, let's optimize this.
            
            # If current_max disappears, the new max is at most current_max - 1 (unless max_new_abs > current_max, which is impossible if we are checking for improvement? No, delta can increase sum).
            # If max_new_abs > current_max, new_max = max_new_abs.
            # If max_new_abs <= current_max, and current_max disappears, new_max < current_max.
            
            if max_new_abs > current_max:
                new_max = max_new_abs
            else:
                # We need to find the highest k < current_max with hist[k] > 0 (considering updates)
                # Updates:
                # For k in new_abs_vals: hist[k] increases
                # For k in old_abs_vals: hist[k] decreases
                
                # This is getting complex. Let's just do the update and revert.
                pass
        
        # Given the complexity of "predicting" the max without updating,
        # and N=256, let's just do the update, check, and revert if needed.
        
        # Store old values for revert
        revert_data = []
        
        for ap_idx in aps_containing[i]:
            old_sum = ap_sums[ap_idx]
            old_abs = abs(old_sum)
            new_sum = old_sum + delta
            new_abs = abs(new_sum)
            
            revert_data.append((ap_idx, old_sum, old_abs, new_abs))
            
            # Update hist
            hist[old_abs] -= 1
            hist[new_abs] += 1
            
            # Update ap_sums
            ap_sums[ap_idx] = new_sum
            
        # Find new max
        # If hist[current_max] > 0, new_max = current_max
        # Else, find max k where hist[k] > 0
        
        if hist[current_max] > 0:
            new_max = current_max
        else:
            # Scan down
            new_max = 0
            for k in range(current_max - 1, -1, -1):
                if hist[k] > 0:
                    new_max = k
                    break
            # Also check if any new_abs > current_max? 
            # We know max_new_abs <= current_max in this branch? 
            # No, we didn't check. 
            # If max_new_abs > current_max, then hist[max_new_abs] would be > 0.
            # But we scanned down from current_max - 1.
            # So we must check max_new_abs.
            
            # Let's just scan the whole hist? O(N) = 256.
            # 256 operations is fine.
            new_max = 0
            for k in range(N, -1, -1):
                if hist[k] > 0:
                    new_max = k
                    break
        
        # Decide acceptance
        accepted = False
        if new_max < current_max:
            accepted = True
        elif new_max == current_max:
            # Accept with probability 0.5 to explore
            if random.random() < 0.5:
                accepted = True
        # else: reject
        
        if not accepted:
            # Revert
            for ap_idx, old_sum, old_abs, new_abs in revert_data:
                ap_sums[ap_idx] = old_sum
                hist[new_abs] -= 1
                hist[old_abs] += 1
            signs[i] = old_sign # No change to signs
            return False, current_max
            
        # If accepted, signs is already flipped? 
        # Wait, I didn't flip signs[i] in the update loop.
        signs[i] = new_sign
        return True, new_max

    best_signs = None
    best_val = N + 1

    # Multi-restart
    # Use different seeds for random init
    
    while time.time() < deadline - 2:
        # Random Init
        signs = [1 if random.random() < 0.5 else -1 for _ in range(N)]
        ap_sums, hist, current_max = compute_state(signs)
        
        if current_max < best_val:
            best_val = current_max
            best_signs = signs.copy()
            if best_val == 1:
                break

        # Local Search
        # We can use a "tabu" list to avoid oscillation?
        # Or just random walk.
        
        # To escape local minima, we can allow worsening moves with low prob?
        # Or we can just restart.
        
        # Let's run local search for a fixed number of steps or time slice
        steps = 0
        max_steps = 50000 # Adjust based on speed
        
        while steps < max_steps and time.time() < deadline - 2:
            i = random.randint(0, N - 1)
            accepted, new_max = try_flip(signs, i, ap_sums, hist, current_max)
            if accepted:
                current_max = new_max
                if current_max < best_val:
                    best_val = current_max
                    best_signs = signs.copy()
                    if best_val == 1:
                        break
            steps += 1
            
        if best_val == 1:
            break

    if best_signs is None:
        return [1] * N
        
    return best_signs
