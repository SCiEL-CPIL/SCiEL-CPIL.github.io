"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from collections import defaultdict

def _isqrt(x):
    return math.isqrt(x)

def _sigma_sieve_numpy(n):
    """Compute sigma(k) for k=1..n using NumPy vectorization."""
    sigma = np.zeros(n + 1, dtype=np.int64)
    for i in range(1, n + 1):
        # Add i to all multiples of i
        sigma[i::i] += i
    return sigma

def _sigma_sieve_python(n):
    """Compute sigma(k) for k=1..n using a pure Python sieve (fallback)."""
    sigma = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(i, n + 1, i):
            sigma[j] += i
    return sigma

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline, 20s safety margin
    
    # Start with a known solution
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = len(best_k)
    
    # Strategy: Compute sigma for a large N once, then group by M value.
    # The numpy sieve is O(N log N) with good constant.
    # We'll try progressively larger N, but only recompute if we have time.
    
    # Try a single large computation first
    for N in [5 * 10**6, 10**7, 2 * 10**7, 5 * 10**7]:
        elapsed = time.time() - start_time
        # Estimate: sieve for N takes roughly N * log(N) / 1e8 seconds (rough heuristic)
        # For N=5e6, ~5e6*15/1e8 = 0.75s; N=1e7, ~1e7*16/1e8 = 1.6s
        # Be conservative: allow 80% of remaining time for the sieve
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        try:
            sigma = _sigma_sieve_numpy(N)
        except Exception:
            break
        
        elapsed_sieve = time.time() - start_time
        if time.time() > deadline - 10:
            break
        
        # Group by M value using a dictionary
        groups = defaultdict(list)
        sigma_list = sigma.tolist()
        
        # Process in chunks to check time
        chunk_size = 10**6
        for k_start in range(1, N + 1, chunk_size):
            k_end = min(k_start + chunk_size, N + 1)
            for k in range(k_start, k_end):
                M_val = k * sigma_list[k]
                if M_val <= 10**18:
                    # k <= sqrt(M) is always true for k >= 1 since sigma(k) >= k
                    groups[M_val].append(k)
            
            if time.time() > deadline - 15:
                break
        
        # Find the best M with most solutions
        for M_val, ks in groups.items():
            count = len(ks)
            if count > best_count:
                best_M = M_val
                best_k = ks
                best_count = count
        
        if time.time() > deadline - 15:
            break
        
        # If we found a very good solution, stop
        if best_count >= 100:
            break
    
    # Ensure types are correct
    best_M = int(best_M)
    best_k = [int(k) for k in best_k]
    
    # Verify the solution
    for k in best_k:
        if k > _isqrt(best_M):
            return {'M': 333312, 'k': [336, 372, 434]}
    
    return {'M': best_M, 'k': best_k}
# EVOLVE-BLOCK-END
