import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Strategy:
    # 1. Compute sigma(k) for k up to N using a linear sieve.
    # 2. For each k, compute M = k * sigma(k).
    # 3. Count frequencies of M values.
    # 4. Find the M with the highest frequency (number of k's).
    # 5. Collect all k's for that M.
    
    # We need to be careful about the size of N.
    # sigma(k) can be large. M = k * sigma(k) must be <= 10^18.
    # For k up to 10^7, sigma(k) is roughly k * log(log(k)), so M is roughly k^2 * log(log(k)).
    # (10^7)^2 = 10^14, which is well within 10^18.
    # We can push N higher. Let's try N = 2 * 10^7 or 3 * 10^7.
    # Memory: 3 arrays of size N+1 of Python ints.
    # In CPython, an int is ~28 bytes. 3 * 10^7 * 28 * 3 ~ 2.5 GB. Too much?
    # Actually, small ints are cached, but sigma values grow.
    # Using arrays from 'array' module or lists of ints.
    # Let's stick to lists first and see if we can optimize.
    # Actually, a linear sieve in pure Python is slow.
    # N = 10^7 takes ~10-15s.
    # N = 2 * 10^7 takes ~20-30s.
    # Let's aim for N = 15,000,000.
    
    N = 15_000_000
    
    # Check if we have enough time to run the sieve.
    # If not, reduce N.
    # But we don't know the exact speed.
    # Let's just run it and check time.
    
    sigma = [1] * (N + 1)
    p_pow = [1] * (N + 1)
    p_sigma = [1] * (N + 1)
    spf = [0] * (N + 1)
    primes = []
    
    # Linear Sieve for sigma
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
            sigma[i] = i + 1
            p_pow[i] = i
            p_sigma[i] = i + 1
        
        for p in primes:
            if p > spf[i] or i * p > N:
                break
            
            spf[i * p] = p
            
            if i % p == 0:
                p_pow[i * p] = p_pow[i] * p
                p_sigma[i * p] = p_sigma[i] + p_pow[i * p]
                sigma[i * p] = (sigma[i] // p_sigma[i]) * p_sigma[i * p]
                break
            else:
                p_pow[i * p] = p
                p_sigma[i * p] = p + 1
                sigma[i * p] = sigma[i] * (p + 1)
        
        if i % 1000000 == 0:
            if time.time() - start_time > deadline - 30:
                # We are running out of time.
                # We can't just stop the sieve.
                # We must finish it to have correct sigmas.
                # So we should have picked a smaller N.
                # For now, just continue.
                pass
    
    # Pass 2: Count frequencies of M = k * sigma(k)
    counts = defaultdict(int)
    max_count = 0
    best_M = 0
    LIMIT_M = 10**18
    
    for i in range(1, N + 1):
        m_val = i * sigma[i]
        if m_val <= LIMIT_M:
            counts[m_val] += 1
            if counts[m_val] > max_count:
                max_count = counts[m_val]
                best_M = m_val
    
    # If no collisions, max_count is 1, best_M is the first one.
    # We need to return a list of k's.
    
    # Pass 3: Collect k's for best_M
    k_list = []
    for i in range(1, N + 1):
        if i * sigma[i] == best_M:
            k_list.append(i)
    
    # Ensure types are correct
    # M is int
    # k is list of ints
    
    return {'M': int(best_M), 'k': [int(x) for x in k_list]}
