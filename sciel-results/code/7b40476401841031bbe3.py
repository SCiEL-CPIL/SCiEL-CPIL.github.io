import time
import math
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    
    # Increase LIMIT_N to 5 * 10^7
    # Memory: phi (400MB) + last_seen (200MB) + is_prime (50MB) ~ 650MB
    # Time: Sieve ~10-15s, Loop ~5s, NumPy ~2s. Total < 30s.
    LIMIT_N = 5 * 10**7
    
    # 1. Sieve of Eratosthenes using NumPy
    is_prime = np.ones(LIMIT_N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(LIMIT_N**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
            
    # 2. Compute phi using NumPy
    phi = np.arange(LIMIT_N + 1, dtype=np.int64)
    primes = np.where(is_prime)[0]
    
    for p in primes:
        multiples = np.arange(p, LIMIT_N + 1, p)
        phi[multiples] -= phi[multiples] // p
        
    # 3. Compute prev array and find max run
    # last_seen[v] stores the last index where phi[index] == v
    # v is at most LIMIT_N, so we can use an array of size LIMIT_N + 1
    last_seen = np.zeros(LIMIT_N + 1, dtype=np.int32)
    prev = np.zeros(LIMIT_N + 1, dtype=np.int32)
    
    # Fill prev array
    # phi[0] is 0, we start from 1
    for j in range(1, LIMIT_N + 1):
        v = phi[j]
        prev[j] = last_seen[v]
        last_seen[v] = j
        
    # Calculate prefix max of prev
    # prev[0] is 0, we care about 1..LIMIT_N
    prefix_max = np.maximum.accumulate(prev)
    
    # lengths[j-1] is the length of the valid window ending at j
    # j ranges from 1 to LIMIT_N
    # indices 0 to LIMIT_N-1 in lengths
    j_indices = np.arange(1, LIMIT_N + 1, dtype=np.int64)
    lengths = j_indices - prefix_max[1:LIMIT_N+1]
    
    # Find the maximum length
    max_len = np.max(lengths)
    max_idx = np.argmax(lengths)
    
    best_k = int(max_len)
    # max_idx is 0-based index in lengths
    # j = max_idx + 1
    # start = j - best_k + 1
    best_s = int(max_idx + 1 - best_k + 1)
    
    # Sanity check
    if best_s < 1:
        best_s = 1
    if best_s + best_k - 1 > LIMIT_N:
        best_s = LIMIT_N - best_k + 1
        if best_s < 1:
            best_k = LIMIT_N
            best_s = 1

    return {'start': best_s, 'length': best_k}
