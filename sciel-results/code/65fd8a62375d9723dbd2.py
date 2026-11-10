import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit
    
    n = 128
    target_lo = 128
    target_hi = 256
    targets = np.arange(target_lo, target_hi + 1)
    num_targets = len(targets)
    
    # Bitset representation for A
    # A is a set of integers in [0, 128]
    
    def compute_counts(A_mask):
        """
        Compute counts of pair sums for elements in A_mask.
        A_mask is an int where bit i is 1 if i is in A.
        Returns a numpy array of counts for sums 0..256.
        """
        counts = np.zeros(257, dtype=np.int32)
        # Extract elements for faster iteration than checking bits every time?
        # Actually, for N=128, iterating over set bits is fast.
        # But we can also just iterate 0..128.
        for i in range(n + 1):
            if (A_mask >> i) & 1:
                for j in range(i, n + 1):
                    if (A_mask >> j) & 1:
                        counts[i + j] += 1
        return counts

    def get_metrics(counts):
        seg = counts[target_lo:target_hi + 1]
        peak = int(np.max(seg))
        uncovered = int(np.sum(seg == 0))
        ss = int(np.sum(seg ** 2))
        return peak, uncovered, ss

    def count_bit_set(A_mask, i):
        return 1 if (A_mask >> i) & 1 else 0

    def update_counts_remove_add(counts, A_mask, x_out, x_in):
        """
        Efficiently update counts when removing x_out and adding x_in.
        Assumes x_out is in A_mask and x_in is NOT in A_mask.
        """
        new_counts = counts.copy()
        
        # Remove x_out
        # Pairs (x_out, y) for y in A_mask, y >= x_out? 
        # No, we must remove all pairs involving x_out.
        # Since counts are for unordered pairs (i,j) with i<=j:
        # If y < x_out: pair (y, x_out) was counted at index y+x_out. 
        # If y == x_out: pair (x_out, x_out) was counted.
        # If y > x_out: pair (x_out, y) was counted.
        
        # Iterate all y in A_mask
        # We need to be careful: if x_in is also in the range, we add its pairs.
        
        # Remove x_out contributions
        for y in range(n + 1):
            if y == x_out:
                continue
            if (A_mask >> y) & 1:
                # Pair (min(x_out, y), max(x_out, y)) is removed
                # It contributed 1 to counts[x_out + y]
                new_counts[x_out + y] -= 1
        # Pair (x_out, x_out)
        new_counts[2 * x_out] -= 1
        
        # Add x_in contributions
        # Note: A_mask still has x_out set. We need to consider the NEW set.
        # The new set is (A_mask \ {x_out}) U {x_in}.
        # So we add pairs (x_in, y) where y is in (A_mask \ {x_out}) U {x_in}.
        
        # Pairs with y in A_mask (excluding x_out if it was there, but we are adding to new_counts)
        # Actually, it's safer to just iterate y in current A_mask (which has x_out, not x_in)
        # and adjust.
        
        # Let's do it cleanly:
        # New set elements: y such that (y in A_mask and y != x_out) or y == x_in
        
        # Pairs (x_in, y) for y in (A_mask \ {x_out})
        for y in range(n + 1):
            if y == x_out:
                continue
            if (A_mask >> y) & 1:
                # y is in the set without x_out
                # Pair (x_in, y) is added
                # We add 1 to counts[x_in + y]
                new_counts[x_in + y] += 1
                
        # Pair (x_in, x_in)
        new_counts[2 * x_in] += 1
        
        return new_counts

    best_A_mask = 0
    best_peak = float('inf')
    best_ss = float('inf')
    best_A_list = []

    # 1. Greedy Initialization
    # We want a set A that covers all targets [128, 256].
    # Start with empty set, add elements that cover most uncovered targets.
    
    A_mask = 0
    counts = np.zeros(257, dtype=np.int32)
    
    # Pre-compute potential gains? 
    # For each candidate x, and each target t, does x help?
    # x helps t if there exists y in A such that x+y=t (i.e., t-x in A) OR 2x=t.
    
    # We will run multiple random restarts with different initial "seeds"
    
    for restart in range(20):
        if time.time() - start_time > time_limit:
            break
        
        # Random initial small set to break symmetry
        A_mask = 0
        # Pick 5 random elements
        for _ in range(5):
            x = random.randint(0, n)
            A_mask |= (1 << x)
        
        counts = compute_counts(A_mask)
        
        # Greedy fill
        while True:
            if time.time() - start_time > time_limit:
                break
            
            seg = counts[target_lo:target_hi + 1]
            if np.all(seg > 0):
                break
            
            unc_targets = [t for t in targets if counts[t] == 0]
            if not unc_targets:
                break
                
            best_gain = -1
            best_elem = None
            
            # Try all candidates
            for cand in range(n + 1):
                if (A_mask >> cand) & 1:
                    continue
                
                gain = 0
                for t in unc_targets:
                    other = t - cand
                    if 0 <= other <= n:
                        # If other == cand, we need cand in A (which we are adding). 
                        # The pair (cand, cand) covers t.
                        # If other != cand, we need other in A.
                        if other == cand:
                            gain += 1
                        else:
                            if (A_mask >> other) & 1:
                                gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_elem = cand
            
            if best_elem is None or best_gain <= 0:
                # Stuck, break
                break
                
            A_mask |= (1 << best_elem)
            # Recompute counts (slow but safe for init)
            counts = compute_counts(A_mask)
            
        # Check if valid
        peak, unc, ss = get_metrics(counts)
        if unc == 0:
            if peak < best_peak or (peak == best_peak and ss < best_ss):
                best_peak = peak
                best_ss = ss
                best_A_mask = A_mask
                best_A_list = [i for i in range(n+1) if (A_mask >> i) & 1]

        # 2. Local Search on the current A_mask
        curr_A_mask = A_mask
        curr_counts = counts
        curr_peak, curr_unc, curr_ss = get_metrics(curr_counts)
        
        if curr_unc > 0:
            continue
            
        # Local Search: Swap elements to reduce peak/ss
        # We try to swap x_out (in A) with x_in (not in A)
        
        temperature = 1.0
        no_improve = 0
        
        while time.time() - start_time < time_limit:
            if no_improve > 500: # Higher limit for faster steps
                break
                
            improved = False
            
            # Pick random x_out from A
            # Get list of elements in A
            # Optimization: maintain a list? Or just iterate bits?
            # Iterating bits is fast enough for N=128.
            
            # To make it faster, let's pick a random index in the range and check if it's in A.
            # Or better, maintain a list of active elements.
            # For simplicity and speed, let's just pick random candidates for x_in and x_out.
            
            # Pick x_in (not in A)
            x_in = random.randint(0, n)
            if (curr_A_mask >> x_in) & 1:
                continue
                
            # Pick x_out (in A)
            # We need to pick one from the set.
            # Let's pick a random number of bits set?
            # Simpler: Just iterate through a few random candidates for x_out until we find one in A.
            # Or, precompute list of A elements if it's small.
            # Let's assume A is small (< 50).
            
            # Efficient way:
            # temp_mask = curr_A_mask
            # while temp_mask:
            #   ...
            # This is getting complex. Let's stick to random sampling for x_out.
            
            found_x_out = False
            for _ in range(10): # Try 10 times to find an x_out
                x_out = random.randint(0, n)
                if (curr_A_mask >> x_out) & 1:
                    found_x_out = True
                    break
            
            if not found_x_out:
                continue
                
            if x_out == x_in:
                continue
                
            # Evaluate swap
            new_counts = update_counts_remove_add(curr_counts, curr_A_mask, x_out, x_in)
            new_peak, new_unc, new_ss = get_metrics(new_counts)
            
            if new_unc > 0:
                continue
                
            delta_peak = new_peak - curr_peak
            delta_ss = new_ss - curr_ss
            
            # Accept if better
            if delta_peak < 0 or (delta_peak == 0 and delta_ss < 0):
                curr_A_mask = (curr_A_mask & ~(1 << x_out)) | (1 << x_in)
                curr_counts = new_counts
                curr_peak, curr_unc, curr_ss = new_peak, new_unc, new_ss
                improved = True
                no_improve = 0
                
                if curr_peak < best_peak or (curr_peak == best_peak and curr_ss < best_ss):
                    best_peak = curr_peak
                    best_ss = curr_ss
                    best_A_mask = curr_A_mask
                    best_A_list = [i for i in range(n+1) if (curr_A_mask >> i) & 1]
            else:
                # Metropolis acceptance
                delta = delta_peak * 100 + delta_ss
                if delta > 0:
                    prob = np.exp(-delta / max(temperature, 0.01))
                    if random.random() < prob:
                        curr_A_mask = (curr_A_mask & ~(1 << x_out)) | (1 << x_in)
                        curr_counts = new_counts
                        curr_peak, curr_unc, curr_ss = new_peak, new_unc, new_ss
                        improved = True
                        no_improve = 0
            
            if improved:
                temperature *= 0.999
            else:
                no_improve += 1
                temperature *= 0.99
                if temperature < 0.1:
                    # Restart from best
                    curr_A_mask = best_A_mask
                    curr_counts = compute_counts(curr_A_mask)
                    curr_peak, curr_unc, curr_ss = get_metrics(curr_counts)
                    temperature = 1.0
                    no_improve = 0

    if best_A_list:
        return best_A_list
    
    # Fallback
    return sorted(list(range(64, 129)))
