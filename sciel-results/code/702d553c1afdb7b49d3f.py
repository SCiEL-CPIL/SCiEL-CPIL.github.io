import math
import time

def construct():
    start_time = time.time()
    deadline = 110.0 # Leave 10s margin
    
    def sigma_sieve(n):
        sigma = [0] * (n + 1)
        for i in range(1, n + 1):
            for j in range(i, n + 1, i):
                sigma[j] += i
        return sigma

    # We will try increasing N
    best_M = 333312
    best_ks = [336, 372, 434]
    
    # Try N = 2 * 10^6 first
    N = 2000000
    if time.time() - start_time > deadline:
        return {'M': best_M, 'k': best_ks}
        
    try:
        sigma = sigma_sieve(N)
        counts = {}
        ks_map = {}
        
        for k in range(1, N + 1):
            m_val = k * sigma[k]
            if m_val > 10**18:
                continue
            if m_val not in counts:
                counts[m_val] = 0
                ks_map[m_val] = []
            counts[m_val] += 1
            ks_map[m_val].append(k)
            
        # Find best
        current_best_count = len(best_ks)
        for m_val, cnt in counts.items():
            if cnt > current_best_count:
                current_best_count = cnt
                best_M = m_val
                best_ks = ks_map[m_val]
                
    except MemoryError:
        pass
    except Exception:
        pass
        
    # If we have time, try larger N
    if time.time() - start_time < deadline - 20:
        N2 = 5000000
        try:
            sigma = sigma_sieve(N2)
            counts = {}
            ks_map = {}
            
            for k in range(1, N2 + 1):
                m_val = k * sigma[k]
                if m_val > 10**18:
                    continue
                if m_val not in counts:
                    counts[m_val] = 0
                    ks_map[m_val] = []
                counts[m_val] += 1
                ks_map[m_val].append(k)
                
            current_best_count = len(best_ks)
            for m_val, cnt in counts.items():
                if cnt > current_best_count:
                    current_best_count = cnt
                    best_M = m_val
                    best_ks = ks_map[m_val]
        except MemoryError:
            pass
        except Exception:
            pass

    # Verify best_ks
    # The task says distinct k_i. The list from map is distinct by construction.
    # We must ensure k_i <= isqrt(M).
    # Since f(k) = k sigma(k) >= k * k = k^2, we have k <= sqrt(M).
    # So this condition is automatically satisfied.
    
    return {'M': best_M, 'k': best_ks}
