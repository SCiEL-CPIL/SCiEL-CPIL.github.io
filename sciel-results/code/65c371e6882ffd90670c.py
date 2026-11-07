"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict
import array

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s margin for post-processing
    
    max_m = 10**18
    
    # Adaptive N: try large N, fall back if too slow
    # The sieve is O(N log log N) but with Python overhead ~ O(N * avg_prime_divisors)
    # For N=50M, expect ~30-60s in optimized Python
    N = 50000000
    
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
    last_time_check = start_time
    
    for i in range(2, N + 1):
        si = spf[i]
        if si == 0:
            si = i
            spf[i] = i
            primes.append(i)
            p_pow[i] = i
            sigma_p_pow[i] = 1 + i
            sigma[i] = 1 + i
            
        sig_i = sigma[i]
        spp_i = sigma_p_pow[i]
        pp_i = p_pow[i]
        
        for p in primes:
            if p > si:
                break
            ip = i * p
            if ip > N:
                break
            spf[ip] = p
            if p == si:
                pp_ip = pp_i * p
                spp_ip = spp_i + pp_ip
                p_pow[ip] = pp_ip
                sigma_p_pow[ip] = spp_ip
                sigma[ip] = (sig_i // spp_i) * spp_ip
                break
            else:
                p_pow[ip] = p
                sigma_p_pow[ip] = 1 + p
                sigma[ip] = sig_i * (1 + p)
        
        # Time check every 1M iterations to avoid overhead
        if i & 0x3FFFFF == 0:  # every ~4M
            if time.time() > deadline - 30:
                N = i
                break

    m_to_ks = defaultdict(list)
    
    for k in range(1, N + 1):
        s = sigma[k]
        if s:
            M = k * s
            if M <= max_m:
                m_to_ks[M].append(k)

    best_m = 0
    best_count = 0
    for M, ks in m_to_ks.items():
        c = len(ks)
        if c > best_count:
            best_count = c
            best_m = M

    if best_count == 0:
        return {'M': 333312, 'k': [336, 372, 434]}

    best_ks = m_to_ks[best_m]
    sqrt_m = math.isqrt(best_m)
    best_ks = [k for k in best_ks if k <= sqrt_m]
    best_ks.sort()

    if not best_ks:
        return {'M': 333312, 'k': [336, 372, 434]}

    return {'M': int(best_m), 'k': [int(k) for k in best_ks]}
# EVOLVE-BLOCK-END
