import time
import math

def construct():
    start_time = time.time()
    # Safety margin: 100 seconds
    deadline = start_time + 100.0
    
    # Fallback solution
    fallback_M = 333312
    fallback_ks = [336, 372, 434]
    
    # We will search for k in range [1, N]
    # Estimate N based on time.
    # Let's start with N = 2 * 10**6 and see if we can go higher.
    # Or just pick a fixed N that is safe.
    # 10**6 is very safe. 5*10**6 is likely safe.
    # Let's try 5 * 10**6.
    
    N = 5 * 10**6
    
    # Check if we have enough time for a larger N?
    # For now, fixed N to avoid complexity.
    
    # Compute sigma(k) for k = 1..N
    # sigma = [0] * (N + 1)
    # Using a list of ints.
    sigma = [0] * (N + 1)
    
    # Sieve for sigma
    # This is O(N log N)
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma[j] += i
            
    # Check time
    if time.time() > deadline - 10:
        return {'M': fallback_M, 'k': fallback_ks}
        
    counts = {}
    solutions = {}
    
    # Iterate k and compute M
    for k in range(1, N + 1):
        M = k * sigma[k]
        if M > 10**18:
            continue
        if M not in counts:
            counts[M] = 1
            solutions[M] = [k]
        else:
            counts[M] += 1
            solutions[M].append(k)
            
    # Find the M with the most solutions
    max_count = 0
    best_M = fallback_M
    best_ks = fallback_ks
    
    for M, c in counts.items():
        if c > max_count:
            max_count = c
            best_M = M
            best_ks = solutions[M]
            
    # If we found a better solution, return it
    if max_count > len(fallback_ks):
        # Ensure types are correct
        return {'M': int(best_M), 'k': [int(k) for k in best_ks]}
    else:
        return {'M': fallback_M, 'k': fallback_ks}
