import math
import time
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    # We will search up to this limit for k
    # 25 million is a balance between finding more collisions and memory/time
    N_MAX = 25000000

    # Precompute sigma for all k using a divisor-sum sieve with NumPy
    # Use int32 to save memory. Max sigma(25M) is well within 2^31
    sigma_arr = np.zeros(N_MAX + 1, dtype=np.int32)
    
    # Sieve
    # The loop is the bottleneck.
    for d in range(1, N_MAX + 1):
        if d % 5000 == 0:
            if time.time() > deadline - 30:
                break
        sigma_arr[d::d] += d

    # Group by M = k * sigma(k)
    # To save memory, we only store the list of k's for M's that have > 1 solution.
    # We always store the count.
    m_to_count = {}
    m_to_list = {}
    
    best_count = 0
    best_M = 1
    best_ks = [1]

    # Iterate k
    for k in range(1, N_MAX + 1):
        if k % 50000 == 0:
            if time.time() > deadline - 10:
                break

        s = int(sigma_arr[k])
        M = k * s  # Python int

        if M <= 10**18:
            # Update count
            c = m_to_count.get(M, 0) + 1
            m_to_count[M] = c
            
            if c == 1:
                # First time seeing this M, don't store list yet to save memory
                # We only need the list if it becomes a candidate for best
                # But we don't know if it will be best.
                # However, if we don't store it, we can't return it if it IS best.
                # So we must store it if we want to return it.
                # But storing list for EVERY M is expensive.
                
                # Compromise:
                # Store list only if c > 1?
                # If c == 1, we just store the single k in a separate dict?
                # Or just store in m_to_list only if c > 1?
                # If c == 1, we can't update best (best_count >= 1).
                # If c > 1, we update best if c > best_count.
                # So we only need the list for M's with c > 1.
                
                # Wait, if best_count is 0 initially, we need a list for c=1?
                # No, we initialize best with a known solution.
                # Let's assume we find at least one collision.
                
                # If c == 1, we do nothing for the list.
                # If c > 1:
                #   If c == 2, we need to create the list with the previous k and current k.
                #   But we didn't store the previous k!
                
                # So we MUST store the first k for every M.
                # We can store `m_to_first_k[M] = k`.
                # Then if c == 2, `m_to_list[M] = [m_to_first_k[M], k]`.
                # If c > 2, `m_to_list[M].append(k)`.
                
                # Memory: `m_to_first_k` has N entries.
                # `m_to_count` has N entries.
                # `m_to_list` has only collision M's.
                
                # This is better than storing lists for all.
                # But `m_to_first_k` is still N entries.
                
                # Let's use `m_to_first_k`.
                pass 
            else:
                # c > 1
                if c == 2:
                    # We need the first k.
                    # We didn't store it in m_to_list.
                    # We need a way to get it.
                    # Let's change strategy:
                    # Store `m_to_first_k` for all M.
                    # Store `m_to_list` only for M with count > 1.
                    pass

    # Revised Strategy for Memory:
    # m_to_first_k: dict M -> k (first seen)
    # m_to_count: dict M -> count
    # m_to_list: dict M -> list (only if count > 1)
    
    # Let's re-implement the loop with this.
    
    m_to_first_k = {}
    m_to_count = {}
    m_to_list = {}
    
    best_count = 0
    best_M = 1
    best_ks = [1]
    
    for k in range(1, N_MAX + 1):
        if k % 50000 == 0:
            if time.time() > deadline - 10:
                break

        s = int(sigma_arr[k])
        M = k * s

        if M <= 10**18:
            if M not in m_to_count:
                m_to_count[M] = 1
                m_to_first_k[M] = k
            else:
                c = m_to_count[M] + 1
                m_to_count[M] = c
                
                if c == 2:
                    m_to_list[M] = [m_to_first_k[M], k]
                else:
                    m_to_list[M].append(k)
                
                if c > best_count:
                    best_count = c
                    best_M = M
                    best_ks = m_to_list[M]

    # Verify
    def sigma_exact(n):
        if n == 1: return 1
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
        if k * sigma_exact(k) == best_M:
            final_ks.append(k)
            
    if not final_ks:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    return {'M': int(best_M), 'k': [int(k) for k in final_ks]}
