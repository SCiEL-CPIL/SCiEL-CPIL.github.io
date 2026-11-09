import numpy as np
import time

def construct():
    N = 10000000
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin for safety
    
    # 1. Compute sigma(n) for all n <= N
    # Using NumPy for speed
    sigma = np.ones(N + 1, dtype=np.int64)
    
    # Sieve for sigma
    # sigma[n] = sum of divisors
    # Add i to sigma[i], sigma[2i], ...
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    if time.time() > deadline:
        # Fallback to seed if sieve takes too long (unlikely)
        return [11, 6]

    # 2. Search for best pair
    # Iterate b from N down to 1
    # We want to minimize |2a - 3b| / (2b)
    # Let D = |2a - 3b|. Min D is 1.
    # Error E = D / (2b).
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    # We can prune if 1/(2*b) > best_error
    # Because D >= 1, so E >= 1/(2*b).
    # If 1/(2*b) > best_error, then for this b and all smaller b, 
    # the minimum possible error is > best_error.
    
    for b in range(N, 0, -1):
        # Pruning check
        if 1.0 / (2.0 * b) > best_error:
            break
            
        if time.time() > deadline:
            break
            
        sb = sigma[b]
        
        # Target a is approx 1.5 * b
        target_a = 1.5 * b
        a_center = int(round(target_a))
        
        # Check neighbors of a_center in order of increasing |D|
        # D = |2a - 3b|
        # For a = a_center + da, D = |2(a_center+da) - 3b|
        # = |2*a_center - 3b + 2*da|
        # Let base_D = |2*a_center - 3b|
        # Then D(da) = |base_D + 2*da|
        
        # Generate candidate da values sorted by resulting D
        # We check da in range [-20, 20]
        candidates = []
        for da in range(-20, 21):
            a = a_center + da
            if a < 1 or a > N:
                continue
            D = abs(2 * a - 3 * b)
            if D == 0:
                continue
            candidates.append((D, a))
        
        if not candidates:
            continue
        
        # Sort by D
        candidates.sort(key=lambda x: x[0])
        
        min_D = candidates[0][0]
        
        # Check all candidates with D == min_D
        for D, a in candidates:
            if D > min_D:
                break
            if sigma[a] == sb:
                error = D / (2.0 * b)
                if error < best_error:
                    best_error = error
                    best_a = a
                    best_b = b
                break  # Found a match with min_D for this b, move to next b

    return [int(best_a), int(best_b)]
