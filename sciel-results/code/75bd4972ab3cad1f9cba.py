"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 105  # 15s margin before 120s limit
    
    max_m = 10**18
    max_k = int(math.isqrt(max_m))  # k must be <= sqrt(M)
    
    # Primes to consider
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n_primes = len(primes)
    
    # M -> list of k
    m_to_ks = defaultdict(list)
    
    def gen_combos(idx, current_M, current_k):
        if time.time() > deadline:
            return
        
        if current_M > max_m:
            return
        
        if idx >= n_primes:
            if current_M > 1 and current_k >= 1 and current_k <= max_k:
                m_to_ks[current_M].append(current_k)
            return
        
        p = primes[idx]
        # Option 1: don't include this prime (a=0)
        gen_combos(idx + 1, current_M, current_k)
        
        # Option 2: include this prime with various exponents
        p_pow = 1
        sig_p = 1
        for a in range(1, 60):
            p_pow *= p
            if p_pow > max_k // current_k:
                break
            sig_p += p_pow
            L = p_pow * sig_p
            new_M = current_M * L
            if new_M > max_m:
                break
            new_k = current_k * p_pow
            gen_combos(idx + 1, new_M, new_k)
    
    gen_combos(0, 1, 1)
    
    # Find the best M
    best_m = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    for M, ks in m_to_ks.items():
        if len(ks) > best_count:
            best_count = len(ks)
            best_m = M
            best_ks = ks[:1000]
            if best_count >= 1000:
                break
    
    # Ensure types are correct
    best_m = int(best_m)
    best_ks = [int(x) for x in best_ks[:1000]]
    
    return {'M': best_m, 'k': best_ks}
# EVOLVE-BLOCK-END
