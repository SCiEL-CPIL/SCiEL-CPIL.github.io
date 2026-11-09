"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 135  # leave 15s margin before 150s kill
    
    # Fallback known answer
    fallback = {'M': 333312, 'k': [336, 372, 434]}
    
    # Linear sieve for sigma(k) - O(N) time
    # sigma[1] = 1
    # For prime p: sigma[p] = p+1
    # For composite n*p:
    #   if p | n: sigma[n*p] = sigma[n] - sigma[n/pk[n]] * pk[n] + sigma[n/pk[n]] * pk[n] * p
    #     where pk[n] = highest power of spf(n) dividing n
    #     Simplified: sigma[n*p] = sigma[n] // sigma_pk[n] * (sigma_pk[n] + pk[n]*p)
    #   if p |! n: sigma[n*p] = sigma[n] * (p+1)
    
    # Choose N based on available time; start with moderate N
    N = 8_000_000
    
    # Check if we have time for the sieve
    if time.time() - start_time > DEADLINE - 80:
        return fallback
    
    # Linear sieve arrays
    sigma = [0] * (N + 1)
    sigma[1] = 1
    primes = []
    is_comp = bytearray(N + 1)  # more memory efficient
    pk = [1] * (N + 1)       # pk[n] = highest power of spf(n) dividing n
    sigma_pk = [1] * (N + 1)  # sigma_pk[n] = sigma(pk[n])
    
    t0 = time.time()
    for n in range(2, N + 1):
        if not is_comp[n]:
            primes.append(n)
            pk[n] = n
            sigma_pk[n] = n + 1
            sigma[n] = n + 1
        for p in primes:
            prod = n * p
            if prod > N:
                break
            is_comp[prod] = 1
            if n % p == 0:
                pk[prod] = pk[n] * p
                sigma_pk[prod] = sigma_pk[n] + pk[prod]
                sigma[prod] = sigma[n] // sigma_pk[n] * sigma_pk[prod]
                break
            else:
                pk[prod] = p
                sigma_pk[prod] = p + 1
                sigma[prod] = sigma[n] * (p + 1)
        # Periodic time check
        if n % 1000000 == 0:
            if time.time() - start_time > DEADLINE - 20:
                break
    
    if time.time() - start_time > DEADLINE - 10:
        return fallback
    
    # Compute M = k * sigma(k) and find collisions
    M_map = defaultdict(list)
    max_M = 10**18
    for k in range(2, N + 1):
        if time.time() - start_time > DEADLINE - 5:
            break
        M_val = k * sigma[k]
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
