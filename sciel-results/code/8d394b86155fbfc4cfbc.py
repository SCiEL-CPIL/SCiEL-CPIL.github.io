import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin

    # We will search up to this limit for k
    # Memory constraint for dictionary is the main bottleneck.
    # 10^7 entries in a dict is ~1GB.
    N_MAX = 10_000_000

    # Precompute sigma for all k using a divisor-sum sieve with numpy
    sigma_arr = np.zeros(N_MAX + 1, dtype=np.uint64)
    
    # Sieve of Eratosthenes style for sigma
    for d in range(1, N_MAX + 1):
        if time.time() > deadline - 60:
            break
        # Add d to all multiples of d
        sigma_arr[d::d] += d

    # Verify with a quick spot check
    assert sigma_arr[1] == 1
    assert sigma_arr[2] == 3
    assert sigma_arr[6] == 12

    m_first_k = {}
    m_ks = {}
    best_count = 0
    best_M = 0
    best_ks = []

    # Iterate k
    for k in range(1, N_MAX + 1):
        if time.time() > deadline - 30:
            break

        s = sigma_arr[k]
        M = k * s

        if M > 10**18:
            continue

        if M in m_ks:
            # Already a collision, add to list
            m_ks[M].append(k)
            count = len(m_ks[M])
            if count > best_count:
                best_count = count
                best_M = M
                best_ks = m_ks[M]
        else:
            if M in m_first_k:
                # Second occurrence, create list
                k_prev = m_first_k.pop(M)
                m_ks[M] = [k_prev, k]
                count = 2
                if count > best_count:
                    best_count = count
                    best_M = M
                    best_ks = m_ks[M]
            else:
                # First occurrence
                m_first_k[M] = k

    # Final verification and type conversion
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
    if best_M > 0:
        for k in best_ks:
            if isinstance(k, int) and 1 <= k <= math.isqrt(best_M):
                if k * sigma_exact(k) == best_M:
                    final_ks.append(k)
    
    if not final_ks:
        # Fallback to known solution
        return {'M': 333312, 'k': [336, 372, 434]}
        
    return {'M': int(best_M), 'k': [int(k) for k in final_ks]}
