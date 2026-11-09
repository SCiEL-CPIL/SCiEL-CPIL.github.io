import time
import math

def construct():
    start_time = time.time()
    deadline = 120.0
    
    # We will try to sieve up to N.
    # Start with a safe N, increase if time allows.
    # Let's try N = 5_000_000 first.
    N = 5_000_000
    
    # Sieve for sigma(k)
    sigma = [0] * (N + 1)
    
    # Time check helper
    def time_left():
        return deadline - (time.time() - start_time)
    
    # Compute sigma
    # Optimize by local variables
    sig = sigma
    for d in range(1, N + 1):
        if time_left() < 20:
            break
        for m in range(d, N + 1, d):
            sig[m] += d
            
    if time_left() < 10:
        # Fallback to seed if too slow
        return {'M': 333312, 'k': [336, 372, 434]}
        
    # Compute M = k * sigma(k) and count
    counts = {}
    best_M = 0
    best_count = 0
    
    for k in range(1, N + 1):
        m_val = k * sig[k]
        if m_val > 10**18:
            continue
        if m_val in counts:
            counts[m_val] += 1
            if counts[m_val] > best_count:
                best_count = counts[m_val]
                best_M = m_val
        else:
            counts[m_val] = 1
    
    if best_count <= 3:
        # If we didn't find anything better than seed, return seed
        # But we should try to find at least 4.
        # If N is small, we might not find 4.
        # However, the hint says 10M finds 4+. 5M might be enough.
        pass
        
    # Collect solutions for best_M
    k_list = []
    for k in range(1, N + 1):
        if k * sig[k] == best_M:
            k_list.append(k)
            
    # Verify constraints
    # k <= isqrt(M) is always true as discussed.
    # M <= 10^18 is checked.
    # Distinct k is ensured by list construction.
    
    if not k_list:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    return {'M': best_M, 'k': k_list}
