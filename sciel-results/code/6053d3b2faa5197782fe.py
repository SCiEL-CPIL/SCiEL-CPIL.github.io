import random
import time
import math

def construct():
    N = 256
    rng = random.Random(42)
    
    # Precomputation
    # affected[i] will store a list of indices into the 'sums' array
    affected = [[] for _ in range(N)]
    sums = []
    
    # We iterate through all arithmetic progressions defined by start 'a' and step 'd'
    # For each, we generate the prefix sums.
    # A prefix sum of length k for progression (a, d) includes indices a, a+d, ..., a+(k-1)d.
    # If we flip an index 'idx', it affects all prefix sums of any progression (a,d) 
    # such that idx is in the progression, and the prefix length is >= position of idx + 1.
    
    for d in range(1, N):
        for a in range(0, N - d):
            # Determine the indices in this progression
            indices = []
            curr = a
            while curr < N:
                indices.append(curr)
                curr += d
            
            max_len = len(indices)
            
            # Assign IDs to the prefix sums for this (a, d)
            start_id = len(sums)
            for _ in range(max_len):
                sums.append(0)
            
            # Update affected lists
            # For each index at position p (0-indexed) in 'indices',
            # it is part of prefixes of length p+1 to max_len.
            # These correspond to IDs: start_id + p to start_id + max_len - 1.
            for p, idx in enumerate(indices):
                # Append the range of IDs
                # To avoid creating many small lists, we extend
                affected[idx].extend(range(start_id + p, start_id + max_len))
                
    total_sums = len(sums)
    
    # Initialize random sequence
    s = [rng.choice((-1, 1)) for _ in range(N)]
    
    # Initialize sums array
    # sums[lid] = sum of s[j] for j in the prefix corresponding to lid
    # We can do this by iterating j and adding s[j] to all sums in affected[j]
    for j in range(N):
        val = s[j]
        for lid in affected[j]:
            sums[lid] += val
            
    # Initialize histogram of absolute sums? 
    # We need to track max absolute sum.
    # Sums are in [-N, N].
    # We'll use a histogram of the actual sums to find max abs efficiently.
    # Hist size 2*N + 1. Index v + N.
    hist_size = 2 * N + 1
    offset = N
    hist = [0] * hist_size
    
    for val in sums:
        hist[val + offset] += 1
        
    # Find initial max
    current_max = 0
    for v in range(N, -1, -1):
        # Check v and -v
        if hist[v + offset] > 0 or (v > 0 and hist[-v + offset] > 0):
            current_max = v
            break
            
    # Simulated Annealing
    start_time = time.time()
    T = 1.0
    cool_rate = 0.999
    min_T = 0.01
    
    # To speed up, we can pre-compute the delta values? No, delta is always +/- 2.
    
    while time.time() - start_time < 55:
        i = rng.randint(0, N - 1)
        
        old_s = s[i]
        new_s = -old_s
        delta = new_s - old_s # +/- 2
        
        s[i] = new_s
        
        # Update sums and histogram
        # We need to revert if rejected
        # Store old values? No, just revert by subtracting delta.
        
        # We need to track if current_max needs updating.
        # It's easier to just update hist, then recalculate max if necessary.
        
        for lid in affected[i]:
            old_val = sums[lid]
            new_val = old_val + delta
            
            # Update hist
            hist[old_val + offset] -= 1
            hist[new_val + offset] += 1
            
            sums[lid] = new_val
            
        # Determine new max
        # If any new_val has abs > current_max, update.
        # Else, if the bin for current_max (and -current_max) is empty, decrement.
        
        # Optimization: Check if current_max is still valid
        # Check bins for current_max and -current_max
        # Note: if current_max is 0, check bin 0.
        
        # It's possible that a new_val exceeded current_max.
        # We can check the max of the updated values?
        # There are ~1400 updated values.
        # max_updated = max(abs(v) for v in updated_vals)
        # new_max = max(current_max, max_updated)
        # Then check if new_max is valid in hist.
        
        # Let's collect updated values? Or just track max during update.
        max_updated_abs = 0
        # Re-do update loop to track max? No, we already did it.
        # We can just compute max from the affected lids?
        # That's another loop.
        
        # Alternative: Just check if hist[current_max + offset] == 0 and hist[-current_max + offset] == 0 (if current_max > 0).
        # If so, decrement.
        # But what if a new value is larger?
        # We can check: if any hist[k] > 0 for k > current_max.
        # Since we only changed ~1400 values, the new max can only be:
        # 1. The old max (if still present)
        # 2. One of the new values.
        
        # So:
        # candidate_max = current_max
        # for lid in affected[i]:
        #     if abs(sums[lid]) > candidate_max:
        #         candidate_max = abs(sums[lid])
        
        # This is ~1400 ops.
        
        for lid in affected[i]:
            v = abs(sums[lid])
            if v > candidate_max:
                candidate_max = v
                
        # Now check if candidate_max is actually in the histogram
        # It might be that candidate_max was current_max, but it got removed, 
        # and no new value reached it.
        # So we need to verify candidate_max exists.
        
        while candidate_max > 0:
            if hist[candidate_max + offset] > 0 or hist[-candidate_max + offset] > 0:
                break
            candidate_max -= 1
            
        if candidate_max == 0:
             if hist[offset] > 0:
                 candidate_max = 0
             else:
                 # Should not happen, sums always exist
                 candidate_max = 0 
                 
        new_max = candidate_max
        
        if new_max <= current_max:
            current_max = new_max
            # Accept
        else:
            if rng.random() < math.exp(-(new_max - current_max) / T):
                current_max = new_max
                # Accept
            else:
                # Revert
                s[i] = old_s
                for lid in affected[i]:
                    old_val = sums[lid]
                    new_val = old_val - delta # Revert
                    hist[old_val + offset] -= 1
                    hist[new_val + offset] += 1
                    sums[lid] = new_val
                # current_max remains unchanged (should be the old one)
                # We don't need to recalculate max for revert if we trust the logic, 
                # but to be safe, we can assume it's back to previous state.
                
        T *= cool_rate
        if T < min_T:
            T = min_T
            
    return s
