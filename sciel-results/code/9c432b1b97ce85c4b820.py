import time
import bisect
from collections import defaultdict

def construct():
    start = time.time()
    deadline = start + 85.0
    N = 10_000_000

    # Compute sigma(n) for all n up to N using a sieve-like approach
    # sigma[n] = sum of divisors of n
    # We use the identity: sigma(n) = sum_{d|n} d
    # For each d, add d to all multiples of d
    
    import numpy as np
    
    # Use int64 to be safe (sigma can be large)
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # This is the main sieve loop. For d from 1 to N, add d to sigma[d], sigma[2d], sigma[3d], ...
    # Using numpy's fancy indexing for speed
    for d in range(1, N + 1):
        sigma[d::d] += d
        if d % 50000 == 0:
            if time.time() > deadline - 5.0:
                break
    
    # Check if we finished the sieve
    sieve_complete = (d == N) or (time.time() > deadline - 5.0 and d < N)
    
    if time.time() > deadline - 3.0:
        # Not enough time to complete; return a hardcoded good pair
        # Find a known pair with sigma(a) = sigma(b) and a/b close to 3/2
        # 58, 40: sigma(58)=1+2+29+58=90, sigma(40)=1+2+4+5+8+10+20+40=90. 58/40=1.45, error=0.05
        return [58, 40]
    
    # Group numbers by their sigma value
    # Only include numbers for which sigma was fully computed
    max_n_computed = d if not sieve_complete else N
    if max_n_computed < 10000:
        return [58, 40]
    
    groups = defaultdict(list)
    for i in range(1, max_n_computed + 1):
        s = int(sigma[i])
        groups[s].append(i)
    
    if time.time() > deadline - 2.0:
        return [58, 40]
    
    # Find the best pair (a, b) with sigma(a) = sigma(b), a != b, minimizing |a/b - 3/2|
    # |a/b - 3/2| = |2a - 3b| / (2b)
    # We want to minimize this. For a given b, we want a close to 3b/2.
    
    best_a, best_b = 58, 40
    best_num = abs(2 * 58 - 3 * 40)  # = 4
    best_den = 2 * 40  # = 80
    # error = 4/80 = 0.05, score = 1/0.05 = 20
    
    bisect_left = bisect.bisect_left
    
    for lst in groups.values():
        if time.time() > deadline - 1.0:
            break
        l_len = len(lst)
        if l_len < 2:
            continue
        
        # lst is already sorted since we iterate i in order
        for b in lst:
            target = 1.5 * b
            idx = bisect_left(lst, target)
            
            candidates = []
            if idx < l_len:
                candidates.append(lst[idx])
            if idx > 0:
                candidates.append(lst[idx - 1])
            
            for a in candidates:
                if a == b:
                    continue
                # Check that both are within our computed range
                if a > max_n_computed or b > max_n_computed:
                    continue
                
                num = abs(2 * a - 3 * b)
                if num == 0:
                    continue  # Exact 3/2 ratio is invalid
                
                den = 2 * b
                
                # Compare num/den with best_num/best_den
                if num * best_den < best_num * den:
                    best_num = num
                    best_den = den
                    best_a = a
                    best_b = b
    
    return [int(best_a), int(best_b)]
