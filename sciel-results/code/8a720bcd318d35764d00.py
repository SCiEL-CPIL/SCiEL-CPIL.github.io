import time
import numpy as np
from fractions import Fraction
from collections import defaultdict
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85 second deadline to be safe (limit is 90s)
    
    MAX_N = 10_000_000
    target_num, target_den = 3, 2
    
    # Compute sigma(n) for all n up to MAX_N using a vectorized sieve
    # sigma[n] = sum of all positive divisors of n
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    
    # Sieve: for each d, add d to sigma[d], sigma[2d], sigma[3d], ...
    # Vectorized update: sigma[d::d] += d
    # This is much faster than a python loop over m
    for d in range(1, MAX_N + 1):
        sigma[d::d] += d
        
    if time.time() > deadline - 10:
        # Fallback if sieve took too long (unlikely for 10^7 with numpy, but safe)
        return [11, 6]
    
    # Group numbers by their sigma value
    # We want to find pairs (a, b) with sigma[a] == sigma[b]
    # and minimize |a/b - 3/2|
    
    # To save memory and time, we can just iterate through the array and build groups
    # However, building a dict of lists for 10^7 items might be slow/memory heavy.
    # Alternative: Sort indices by sigma value?
    # Or just use defaultdict(list).
    
    sigma_groups = defaultdict(list)
    # Iterate over valid indices 1..MAX_N
    # Using numpy array iteration is slow, so we stick to python loop or vectorized grouping?
    # Python loop over 10^7 is ~0.5-1s. Acceptable.
    for n in range(1, MAX_N + 1):
        sigma_groups[sigma[n]].append(n)
        
    if time.time() > deadline - 5:
        return [11, 6]
        
    best_error = float('inf')
    best_pair = (11, 6)
    best_a = 11
    best_b = 6
    
    # For each group, we have a sorted list of n (since we appended in increasing order)
    # For each b in group, we want a in group such that a/b ~ 3/2
    # a ~ 1.5 * b
    # We can binary search for 1.5*b in the group
    
    for sigma_val, group in sigma_groups.items():
        if len(group) < 2:
            continue
            
        # Check time periodically
        if time.time() > deadline - 2:
            break
            
        # For each b in group, find closest a in group to 1.5*b
        # To optimize, we can just check a few candidates around the binary search position
        # Since the group is sorted, bisect is O(log L)
        
        # We can optimize by noting that if we iterate b, we are doing O(L log L) work per group.
        # Total work is sum(L_g log L_g).
        # Most groups are small. Large groups are rare.
        
        for b in group:
            target_a_float = 1.5 * b
            # Find position in group
            pos = bisect.bisect_left(group, int(target_a_float))
            
            # Check neighbors: pos-1, pos, pos+1
            # We want to minimize |a/b - 1.5| = |2a - 3b| / (2b)
            # Since b is fixed for this inner loop step, minimizing |2a - 3b| is sufficient
            # among candidates.
            
            candidates = []
            if pos > 0:
                candidates.append(group[pos-1])
            if pos < len(group):
                candidates.append(group[pos])
            if pos + 1 < len(group):
                candidates.append(group[pos+1])
                
            for a in candidates:
                if a == b:
                    continue
                # Check if a/b is exactly 3/2 -> 2a == 3b
                if 2 * a == 3 * b:
                    continue
                # Calculate error
                # error = |a/b - 3/2| = |2a - 3b| / (2b)
                err_num = abs(2 * a - 3 * b)
                err_den = 2 * b
                err = err_num / err_den
                
                if err < best_error:
                    best_error = err
                    best_a = a
                    best_b = b
                    best_pair = (a, b)
                    
    if best_error == float('inf'):
        return [11, 6]
        
    return [int(best_a), int(best_b)]
