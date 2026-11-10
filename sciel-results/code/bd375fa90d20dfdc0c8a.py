import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    # Structural change: Use a divisor sieve to compute sigma(n) for all n up to N.
    # This is O(N log N) but with a small constant, enabling us to scan millions of k values.
    # The previous approach did individual factorization, limiting us to ~10^4 candidates.
    
    # Choose N based on time budget
    # Sieve does sum_{d=1}^{N} N/d ≈ N * ln(N) operations
    # For N=5M: ~5M * 15.4 ≈ 77M ops, ~4-6s in Python
    # For N=10M: ~10M * 16.1 ≈ 161M ops, ~8-12s in Python
    # Let's aim for N=5M to be safe
    
    N = 5000000
    
    # Sieve for sigma: for each divisor d, add d to sigma[m] for all multiples m of d
    sigma = [0] * (N + 1)
    for d in range(1, N + 1):
        for m in range(d, N + 1, d):
            sigma[m] += d
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 20:
        # Not enough time for collection phase, return best effort
        # We still have sigma computed, let's do a quick scan
        pass
    
    # Collect M = k * sigma(k) for all k where M <= 10^18 and k <= isqrt(M)
    m_to_k = defaultdict(list)
    
    for k in range(1, N + 1):
        if time.time() > deadline:
            break
        
        sig = sigma[k]
        M = k * sig
        
        if M <= 10**18:
            # Verify k <= isqrt(M)
            if k <= math.isqrt(M):
                m_to_k[M].append(k)
    
    # Find the M with the most distinct k's
    best_M = None
    best_k_list = []
    
    for M, k_list in m_to_k.items():
        if len(k_list) > len(best_k_list):
            best_k_list = k_list
            best_M = M
    
    if not best_M or not best_k_list:
        # Fallback to known small solution
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Verify the best solution strictly
    valid_k = []
    for k in best_k_list:
        if k * sigma[k] == best_M and k <= math.isqrt(best_M):
            valid_k.append(k)
    
    if len(valid_k) > 0:
        return {'M': best_M, 'k': valid_k}
    else:
        # Fallback
        return {'M': 333312, 'k': [336, 372, 434]}
