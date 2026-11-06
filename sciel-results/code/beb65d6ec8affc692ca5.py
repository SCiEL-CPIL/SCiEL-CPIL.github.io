import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Sieve for primes up to 10000
    limit = 10000
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.array([i for i, v in enumerate(is_prime) if v])
    N = len(primes)
    
    if N == 0:
        return []
        
    # dp_len[i] = max length of valid non-decreasing gap sequence ending at prime i
    # dp_gap[i] = min last gap among all max-length sequences ending at prime i
    # parent[i] = index of previous prime in the best sequence ending at i
    INF = 10**9
    
    dp_len = np.ones(N, dtype=np.int64)  # length 1 for each prime
    dp_gap = np.zeros(N, dtype=np.int64)  # gap 0 for length 1 sequences
    parent = np.full(N, -1, dtype=np.int64)
    
    best_len = 1
    best_index = 0
    
    for i in range(1, N):
        if time.time() > deadline:
            break
            
        # Try all previous primes j < i
        for j in range(i):
            gap = primes[i] - primes[j]
            # Can we extend the sequence ending at j?
            # Need: last gap of best sequence ending at j <= gap
            if dp_gap[j] <= gap:
                new_len = dp_len[j] + 1
                # Update dp for i if this gives a longer sequence,
                # or same length but smaller last gap
                if new_len > dp_len[i] or (new_len == dp_len[i] and gap < dp_gap[i]):
                    dp_len[i] = new_len
                    dp_gap[i] = gap
                    parent[i] = j
                    if new_len > best_len:
                        best_len = new_len
                        best_index = i
    
    # Reconstruct
    if best_len == 1:
        return [int(primes[best_index])]
        
    path = []
    curr_idx = best_index
    while curr_idx != -1:
        path.append(int(primes[curr_idx]))
        curr_idx = parent[curr_idx]
        
    path.reverse()
    return path
