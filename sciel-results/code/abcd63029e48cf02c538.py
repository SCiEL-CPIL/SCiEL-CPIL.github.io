import time
import random
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    
    # We will scan up to LIMIT_N
    # 10^7 is a good balance between time and coverage
    LIMIT_N = 10**7
    
    # 1. Sieve of Eratosthenes using NumPy
    # is_prime array
    is_prime = np.ones(LIMIT_N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(LIMIT_N**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    # 2. Compute phi using NumPy
    phi = np.arange(LIMIT_N + 1, dtype=np.int64)
    
    # Get primes
    primes = np.where(is_prime)[0]
    
    for p in primes:
        # Multiples of p
        multiples = np.arange(p, LIMIT_N + 1, p)
        phi[multiples] -= phi[multiples] // p
        
    # 3. Scan for longest run of distinct phi values
    # Using a sliding window with a count dictionary
    # Note: phi values are up to LIMIT_N, so we could use an array for counts
    # but a dict is safer if phi values were larger, though here they are bounded.
    # Actually, an array of size LIMIT_N+1 is 40MB, which is fine.
    # But dict is fine too.
    
    counts = defaultdict(int)
    left = 0
    best_k = 0
    best_s = 1
    
    # We iterate over the phi array
    # phi[0] is 0, phi[1] is 1.
    # We care about s >= 1, so indices 1 to LIMIT_N.
    # However, the run can start at 1.
    
    for right in range(1, LIMIT_N + 1):
        v = phi[right]
        counts[v] += 1
        
        # If we have a duplicate, shrink window from left
        while counts[v] > 1:
            counts[phi[left]] -= 1
            left += 1
            
        current_k = right - left + 1
        if current_k > best_k:
            best_k = current_k
            best_s = left + 1  # Convert to 1-based index for 'start'
            
        # Early termination if we hit max possible k
        if best_k == 200:
            break
            
        # Time check
        if time.time() - start_time > DEADLINE:
            break

    # 4. Random search for larger s (optional, but good for safety)
    # If best_k is small, we might want to search higher.
    # But given the nature of phi, small s usually contain the densest clusters of "bad" (colliding) or "good" (distinct) numbers.
    # Actually, collisions are more likely for small numbers?
    # phi(n) = n(1-1/p)...
    # For large n, phi(n) is roughly n. Collisions phi(n)=phi(m) imply n/m is close to 1.
    # It is generally believed that the longest runs are found in smaller ranges.
    
    # Let's do a few random checks if time remains
    if time.time() - start_time < DEADLINE - 10:
        # Reuse a simple factorization for large numbers
        # But we already have a good best_k. 
        # Random search is unlikely to beat 25 if it exists in 10^7.
        pass

    return {'start': best_s, 'length': best_k}
