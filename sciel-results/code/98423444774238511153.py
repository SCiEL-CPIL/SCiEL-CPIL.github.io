import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    max_m = 10**18
    
    # Primes to use
    primes = [2, 3, 5, 7, 11, 13, 17, 19]
    
    def generate_combinations(prime_list):
        results = [(1, 1)]
        for p in prime_list:
            new_results = []
            p_pow = 1
            sigma_p = 1
            for e in range(30):
                for k, sigma_k in results:
                    new_k = k * p_pow
                    if new_k > 10**9:
                        break
                    new_sigma = sigma_k * sigma_p
                    if new_k * new_sigma <= max_m:
                        new_results.append((new_k, new_sigma))
                p_pow *= p
                if p_pow > 10**9:
                    break
                sigma_p += p_pow
            results = new_results
        return results

    left_primes = primes[:4]
    right_primes = primes[4:]
    
    left_combos = generate_combinations(left_primes)
    right_combos = generate_combinations(right_primes)
    
    # Sort right_combos by M = k*sigma
    # This might help with pruning if we had a limit, but we check in loop
    # Actually, just iterating is fine.
    
    m_to_ks = defaultdict(list)
    
    count = 0
    for k_l, s_l in left_combos:
        if time.time() > deadline:
            break
        m_l = k_l * s_l
        if m_l > max_m:
            continue
        for k_r, s_r in right_combos:
            count += 1
            if count % 1000000 == 0 and time.time() > deadline:
                break
            m_r = k_r * s_r
            m = m_l * m_r
            if m > max_m:
                continue
            k = k_l * k_r
            if k * k <= m: # k <= sqrt(M)
                m_to_ks[m].append(k)
    
    # Find best M
    best_m = 0
    best_ks = []
    for M, ks in m_to_ks.items():
        if len(ks) > len(best_ks):
            best_m = M
            best_ks = ks
    
    if not best_ks:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    best_ks = [int(k) for k in best_ks]
    
    return {'M': int(best_m), 'k': best_ks}
