"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict
from array import array

def construct():
    start_time = time.time()
    DEADLINE = 145  # leave 5s margin before 150s kill
    
    # Fallback known answer
    fallback = {'M': 333312, 'k': [336, 372, 434]}
    
    # We use a linear sieve to compute sigma(k) for k up to N.
    # N=50,000,000 is a good balance of memory and search space.
    # Memory: 3 arrays of 8 bytes * 50M = 1.2GB. Plus dict.
    # If this OOMs, it fails. But 50M is usually safe on modern 2GB+ envs.
    # To be safer, we could use 30M, but 50M finds more collisions.
    N = 50_000_000
    
    # Check if we have time for the sieve
    if time.time() - start_time > DEADLINE - 80:
        return fallback
    
    # Linear sieve arrays using array('q') for 64-bit integers
    sigma = array('q', [0]) * (N + 1)
    sigma[1] = 1
    primes = []
    is_comp = bytearray(N + 1)
    pk = array('q', [1]) * (N + 1)
    sigma_pk = array('q', [1]) * (N + 1)
    
    # Optimization: Local variables for speed
    sigma_arr = sigma
    pk_arr = pk
    sigma_pk_arr = sigma_pk
    is_comp_arr = is_comp
    primes_list = primes
    
    t0 = time.time()
    for n in range(2, N + 1):
        if not is_comp_arr[n]:
            primes_list.append(n)
            pk_arr[n] = n
            sigma_pk_arr[n] = n + 1
            sigma_arr[n] = n + 1
        
        for p in primes_list:
            prod = n * p
            if prod > N:
                break
            is_comp_arr[prod] = 1
            if n % p == 0:
                pk_prod = pk_arr[n] * p
                pk_arr[prod] = pk_prod
                sigma_pk_prod = sigma_pk_arr[n] + pk_prod
                sigma_pk_arr[prod] = sigma_pk_prod
                sigma_arr[prod] = sigma_arr[n] // sigma_pk_arr[n] * sigma_pk_prod
                break
            else:
                pk_arr[prod] = p
                sigma_pk_arr[prod] = p + 1
                sigma_arr[prod] = sigma_arr[n] * (p + 1)
        
        # Periodic time check
        if n % 1_000_000 == 0:
            if time.time() - start_time > DEADLINE - 20:
                return fallback
    
    if time.time() - start_time > DEADLINE - 10:
        return fallback
    
    # Pass 1: Count occurrences of each M
    # We do not store the list of k's to save memory
    M_counts = defaultdict(int)
    max_M = 10**18
    sigma_arr = sigma
    for k in range(1, N + 1):
        if time.time() - start_time > DEADLINE - 5:
            return fallback
        M_val = k * sigma_arr[k]
        if M_val <= max_M:
            M_counts[M_val] += 1
    
    # Find M with most k's
    best_M = fallback['M']
    best_count = len(fallback['k'])
    
    for M_val, cnt in M_counts.items():
        if cnt > best_count:
            best_count = cnt
            best_M = M_val
            if best_count >= 1000:
                break
    
    if time.time() - start_time > DEADLINE - 2:
        return fallback
    
    # Pass 2: Retrieve the k's for the best M
    best_k = []
    for k in range(1, N + 1):
        if k * sigma_arr[k] == best_M:
            best_k.append(k)
            if len(best_k) >= 1000:
                break
        
        if time.time() - start_time > DEADLINE - 1:
            return fallback

    # Ensure types are correct
    best_M = int(best_M)
    best_k = [int(k) for k in best_k[:1000]]
    
    return {'M': best_M, 'k': best_k}
# EVOLVE-BLOCK-END
