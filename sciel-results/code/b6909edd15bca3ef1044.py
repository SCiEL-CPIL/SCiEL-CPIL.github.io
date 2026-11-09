import time
from math import isqrt
from collections import defaultdict
from array import array

def construct():
    start_time = time.time()
    DEADLINE = 145
    
    fallback = {'M': 333312, 'k': [336, 372, 434]}
    
    # Increase N, but keep memory safe.
    # 80M * (8 + 1 + 1) bytes = 800MB.
    N = 80_000_000
    
    if time.time() - start_time > DEADLINE - 80:
        return fallback

    # 1. Compute is_square_free
    limit = isqrt(N)
    is_sq_free = bytearray(b'\x01') * (N + 1)
    is_sq_free[0] = 0
    is_sq_free[1] = 1
    
    # Sieve for primes up to limit
    is_prime_small = bytearray(b'\x01') * (limit + 1)
    if limit >= 2:
        is_prime_small[0] = 0
        is_prime_small[1] = 0
    primes_small = []
    for i in range(2, limit + 1):
        if is_prime_small[i]:
            primes_small.append(i)
            for j in range(i * i, limit + 1, i):
                is_prime_small[j] = 0
                
    # Mark non-square-free numbers
    for p in primes_small:
        p2 = p * p
        for j in range(p2, N + 1, p2):
            is_sq_free[j] = 0
            
    if time.time() - start_time > DEADLINE - 40:
        return fallback

    # 2. Linear Sieve for sigma on square-free numbers
    sigma = array('q', [0]) * (N + 1)
    sigma[1] = 1
    primes = []
    is_comp = bytearray(N + 1)
    
    sigma_arr = sigma
    is_comp_arr = is_comp
    is_sq_arr = is_sq_free
    primes_list = primes
    
    t0 = time.time()
    for n in range(2, N + 1):
        if not is_sq_arr[n]:
            continue
            
        if not is_comp_arr[n]:
            primes_list.append(n)
            sigma_arr[n] = n + 1
            
        for p in primes_list:
            prod = n * p
            if prod > N:
                break
                
            if n % p == 0:
                # prod is not square-free, skip
                break
            else:
                is_comp_arr[prod] = 1
                sigma_arr[prod] = sigma_arr[n] * (p + 1)
                
        if n % 1_000_000 == 0:
            if time.time() - start_time > DEADLINE - 20:
                return fallback

    if time.time() - start_time > DEADLINE - 10:
        return fallback

    # 3. Count collisions
    M_counts = defaultdict(int)
    max_M = 10**18
    for k in range(1, N + 1):
        if not is_sq_arr[k]:
            continue
        M_val = k * sigma_arr[k]
        if M_val <= max_M:
            M_counts[M_val] += 1
            
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

    # 4. Retrieve k's
    best_k = []
    for k in range(1, N + 1):
        if not is_sq_arr[k]:
            continue
        if k * sigma_arr[k] == best_M:
            best_k.append(k)
            if len(best_k) >= 1000:
                break
                
        if time.time() - start_time > DEADLINE - 1:
            return fallback

    best_M = int(best_M)
    best_k = [int(k) for k in best_k[:1000]]
    
    return {'M': best_M, 'k': best_k}
