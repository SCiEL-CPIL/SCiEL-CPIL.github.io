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
    
    # Linear sieve for sigma(k) - O(N) time
    # Use array module to reduce memory footprint, allowing larger N
    N = 30_000_000
    
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
        if n % 1000000 == 0:
            if time.time() - start_time > DEADLINE - 20:
                break
    
    if time.time() - start_time > DEADLINE - 10:
        return fallback
    
    # Compute M = k * sigma(k) and find collisions
    M_map = defaultdict(list)
    max_M = 10**18
    sigma_arr = sigma
    for k in range(2, N + 1):
        if time.time() - start_time > DEADLINE - 5:
            break
        M_val = k * sigma_arr[k]
        if M_val <= max_M:
            M_map[M_val].append(k)
    
    # Find M with most k's where k <= sqrt(M)
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    for M_val, k_list in M_map.items():
        sqrt_M = isqrt(M_val)
        valid_k = [k for k in k_list if k <= sqrt_M]
        if len(valid_k) > best_count:
            best_count = len(valid_k)
            best_M = M_val
            best_k = valid_k
            if best_count >= 1000:
                break
    
    if time.time() - start_time > DEADLINE:
        return fallback
    
    # Ensure types are correct
    best_M = int(best_M)
    best_k = [int(k) for k in best_k[:1000]]
    
    return {'M': best_M, 'k': best_k}
# EVOLVE-BLOCK-END
