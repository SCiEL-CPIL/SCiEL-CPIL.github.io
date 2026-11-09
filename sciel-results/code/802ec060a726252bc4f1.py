import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 110.0
    
    # Search limit
    # 50 million is a good balance of coverage and time
    N = 50_000_000
    
    # 1. Compute phi using sieve
    # Use int32 to save memory and improve cache performance
    phi = np.arange(1, N + 1, dtype=np.int32)
    
    # Sieve for totient
    # Iterate up to sqrt(N)
    limit = int(np.sqrt(N)) + 1
    for i in range(2, limit):
        if phi[i-1] == i: # i is prime (0-indexed: phi[i-1] corresponds to i)
            # Update multiples of i
            # Indices: i, 2i, 3i, ...
            # In 0-indexed array: i-1, 2i-1, ...
            # Step is i
            # Start index is i-1
            phi[i-1::i] -= phi[i-1::i] // i
            
    # Check time
    if time.time() - start_time > deadline - 10:
        # If sieve took too long (unlikely), reduce N? 
        # For now, assume it's fast.
        pass

    # 2. Find longest run of distinct values
    best_k = 0
    best_s = 1
    seen = set()
    j = 1 # Current right end of window (1-indexed integer value)
    
    # We iterate s from 1 to N
    # To speed up, we can stop if N - s + 1 <= best_k
    for s in range(1, N + 1):
        if N - s + 1 <= best_k:
            break
            
        # Expand j as long as phi(j) is not in seen
        # phi[j-1] is the totient of j
        while j <= N and phi[j-1] not in seen:
            seen.add(phi[j-1])
            j += 1
            
        k = j - s
        if k > best_k:
            best_k = k
            best_s = s
            
        # Remove phi(s) from seen and move s
        seen.remove(phi[s-1])
        
        # Optional: Time check inside loop?
        # If we are close to deadline, we might want to break.
        # But breaking might leave us with a suboptimal result if a longer run is later.
        # However, the probability of finding a significantly longer run later
        # is low if we've already scanned 50M numbers.
        # If time.time() - start_time > deadline - 2:
        #     break

    # Cap k at 200 as per constraints
    if best_k > 200:
        best_k = 200
        
    return {'start': best_s, 'length': best_k}
