import time
import numpy as np
import math

def construct():
    start_time = time.time()
    deadline = start_time + 135

    def run_search(N):
        sigma = np.zeros(N + 1, dtype=np.int64)
        
        for i in range(1, N + 1):
            sigma[i::i] += i
            
        ks = np.arange(1, N + 1, dtype=np.int64)
        M_vals = ks * sigma[1:]
        
        mask = M_vals <= 10**18
        if not np.all(mask):
            M_vals = M_vals[mask]
            ks = ks[mask]
        
        if len(M_vals) == 0:
            return 0, []
        
        unique_M, inverse, counts = np.unique(M_vals, return_inverse=True, return_counts=True)
        
        max_count = int(np.max(counts))
        if max_count == 0:
            return 0, []
            
        max_indices = np.where(counts == max_count)[0]
        best_M_idx = max_indices[0]
        best_M = int(unique_M[best_M_idx])
        
        k_indices = np.where(inverse == best_M_idx)[0]
        best_ks = ks[k_indices].tolist()
        
        sqrt_M = math.isqrt(best_M)
        best_ks = [int(k) for k in best_ks if int(k) <= sqrt_M]
        
        if len(best_ks) == 0:
            return 0, []
        
        return best_M, best_ks

    best_M = 0
    best_ks = []
    best_count = 0
    
    for N in [100000, 500000, 1000000, 5000000, 10000000, 20000000, 40000000]:
        if time.time() > deadline - 5:
            break
        M, ks = run_search(N)
        if len(ks) > best_count:
            best_M, best_ks, best_count = M, ks, len(ks)
            
    if best_count == 0:
        return {'M': 333312, 'k': [336, 372, 434]}

    return {'M': int(best_M), 'k': [int(k) for k in best_ks]}
