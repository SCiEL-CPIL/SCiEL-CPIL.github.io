import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin for 150s limit
    
    # We will search up to this limit for k
    N_MAX = 3000000
    
    # Precompute sigma for all k using a divisor-sum sieve
    # sigma[n] = sum of all divisors of n
    # For each d, add d to sigma[m] for all multiples m of d
    sigma_arr = [0] * (N_MAX + 1)
    for d in range(1, N_MAX + 1):
        for m in range(d, N_MAX + 1, d):
            sigma_arr[m] += d
    
    # Verify with a quick spot check
    # sigma(1)=1, sigma(2)=3, sigma(3)=4, sigma(4)=7, sigma(6)=12
    assert sigma_arr[1] == 1
    assert sigma_arr[2] == 3
    assert sigma_arr[6] == 12
    
    m_to_ks = defaultdict(list)
    best_count = 0
    best_M = 333312
    best_ks = [336, 372, 434]
    
    # Iterate k
    for k in range(1, N_MAX + 1):
        if time.time() > deadline:
            break
        
        s = sigma_arr[k]
        M = k * s
        
        if M <= 10**18:
            if k <= math.isqrt(M):
                m_to_ks[M].append(k)
                
                count = len(m_to_ks[M])
                if count > best_count:
                    best_count = count
                    best_M = M
                    best_ks = m_to_ks[M]
    
    # Final verification and type conversion
    # Re-verify sigma for the final list using exact computation
    def sigma_exact(n):
        if n == 1:
            return 1
        result = 1
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                exp = 0
                p_power = 1
                while temp % p == 0:
                    temp //= p
                    exp += 1
                    p_power *= p
                result *= (p_power * p - 1) // (p - 1)
            p += 1 if p == 2 else 2
        if temp > 1:
            result *= (temp * temp - 1) // (temp - 1)
        return result

    final_ks = []
    for k in best_ks:
        if isinstance(k, int) and 1 <= k <= math.isqrt(best_M):
            if k * sigma_exact(k) == best_M:
                final_ks.append(k)
    
    if not final_ks:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    return {'M': int(best_M), 'k': [int(k) for k in final_ks]}
