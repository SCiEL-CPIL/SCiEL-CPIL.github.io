import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 10000000
    
    # Compute sigma(n) for all n up to N using O(N log N) sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        if time.time() > deadline - 15.0:
            break
        sigma[i::i] += i
    
    # Build a mapping from sigma value to list of indices that have that sigma
    # We'll only consider sigma values up to some reasonable max
    # sigma(n) is at most around 2*n for highly composite, but typically ~n * (small factor)
    # Let's find the max sigma value
    max_sigma = int(sigma.max())
    
    # Group indices by sigma value
    # We'll use a dict: sigma_val -> sorted list of indices
    # But building this for all sigma values up to max_sigma might be memory intensive
    # Instead, we can create a dictionary mapping sigma_val to list of indices
    # But max_sigma could be large (~40 million for n=10 million)
    # Let's use a different approach: for each sigma value, collect the indices
    
    # Actually, let's just build the dict directly
    from collections import defaultdict
    sigma_groups = defaultdict(list)
    for i in range(1, N + 1):
        if time.time() > deadline - 12.0:
            break
        sv = int(sigma[i])
        sigma_groups[sv].append(i)
    
    # Now search for best pair
    # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # For each b, we want to find a such that sigma(a) = sigma(b) and a is close to 1.5*b
    
    # Best error found so far
    best_num = 1
    best_den = 2  # Error 1/2 from (1,1)
    best_a = 1
    best_b = 1
    
    # Iterate b from N down to 1
    # Break when 1/(2b) >= best_num/best_den, i.e., best_den >= 2*b*best_num
    for b in range(N, 0, -1):
        if time.time() > deadline - 5.0:
            break
        
        # Break condition
        if best_den >= 2 * b * best_num:
            break
        
        sb = int(sigma[b])
        group = sigma_groups.get(sb)
        if group is None or len(group) < 2:
            continue
        
        # We want a close to 3b/2
        # Use bisect to find the closest elements
        import bisect
        target = 1.5 * b
        
        # Find insertion point
        pos = bisect.bisect_left(group, target)
        
        # Check neighbors around pos
        # We want to check a few candidates around the target
        # Let's check a window of indices in the group
        # But the group could be large, so we only check a limited number
        # Check at most 20 candidates around the target
        
        candidates = set()
        for offset in range(-10, 11):
            idx = pos + offset
            if 0 <= idx < len(group):
                candidates.add(group[idx])
        
        for a in candidates:
            if a == b or a < 1 or a > N:
                continue
            
            # Compute error
            # |a/b - 3/2| = |2a - 3b| / (2b)
            diff = abs(2 * a - 3 * b)
            if diff == 0:
                # Invalid, skip
                continue
            
            err_num = diff
            err_den = 2 * b
            
            # Compare with best
            # err_num/err_den < best_num/best_den
            # err_num * best_den < best_num * err_den
            if err_num * best_den < best_num * err_den:
                best_num = err_num
                best_den = err_den
                best_a = a
                best_b = b
    
    return [int(best_a), int(best_b)]
