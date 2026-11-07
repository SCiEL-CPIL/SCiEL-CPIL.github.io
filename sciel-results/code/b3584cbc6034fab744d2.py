"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict
import array

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    max_m = 10**18
    # Use 8 million as a balance between time and coverage
    N = 8000000
    
    # Use array to save memory
    # 'Q' is unsigned long long (64-bit)
    # 'I' is unsigned int (32-bit)
    sigma = array.array('Q', [0]) * (N + 1)
    spf = array.array('I', [0]) * (N + 1)
    p_pow = array.array('Q', [0]) * (N + 1)
    sigma_p_pow = array.array('Q', [0]) * (N + 1)
    
    sigma[1] = 1
    p_pow[1] = 1
    sigma_p_pow[1] = 1
    
    primes = []
    
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
            p_pow[i] = i
            sigma_p_pow[i] = 1 + i
            sigma[i] = 1 + i
            
        for p in primes:
            if p > spf[i] or i * p > N:
                break
            ip = i * p
            spf[ip] = p
            if p == spf[i]:
                p_pow[ip] = p_pow[i] * p
                sigma_p_pow[ip] = sigma_p_pow[i] + p_pow[ip]
                sigma[ip] = (sigma[i] // sigma_p_pow[i]) * sigma_p_pow[ip]
                break
            else:
                p_pow[ip] = p
                sigma_p_pow[ip] = 1 + p
                sigma[ip] = sigma[i] * (1 + p)
        
        if time.time() > deadline - 20:
            # If we are running out of time, we break the sieve.
            # This results in incomplete sigma for larger i.
            # However, the loop below will only process up to N.
            # If we break early, sigma for i > current is 0.
            # M = k * 0 = 0. 0 <= max_m.
            # This will add k to m_to_ks[0].
            # This is incorrect.
            # We must ensure we only process valid k's.
            # Better to just let it run if N is safe.
            # 8M should be safe.
            break

    m_to_ks = defaultdict(list)
    
    # Only process up to the index we actually computed
    # If we broke early, we need to know the limit.
    # Since we didn't track the limit, we assume full N if no break.
    # To be safe, we check sigma[k] > 0.
    
    for k in range(1, N + 1):
        s = sigma[k]
        if s == 0:
            continue
        M = k * s
        if M <= max_m:
            m_to_ks[M].append(k)

    best_m = 0
    best_count = 0
    for M, ks in m_to_ks.items():
        if len(ks) > best_count:
            best_count = len(ks)
            best_m = M

    if best_count == 0:
        return {'M': 333312, 'k': [336, 372, 434]}

    best_ks = m_to_ks[best_m]
    best_ks = list(set(best_ks))
    sqrt_m = math.isqrt(best_m)
    best_ks = [k for k in best_ks if k <= sqrt_m]
    best_ks.sort()

    if not best_ks:
        return {'M': 333312, 'k': [336, 372, 434]}

    return {'M': int(best_m), 'k': [int(k) for k in best_ks]}
# EVOLVE-BLOCK-END
