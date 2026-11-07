import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140

    def run_search(N):
        # Use numpy for memory efficiency and speed
        sigma = np.zeros(N + 1, dtype=np.int64)
        
        # Sieve
        # Note: i=1 to N
        for i in range(1, N + 1):
            sigma[i::i] += i
            
        # Compute M = k * sigma(k)
        # k is 1 to N
        ks = np.arange(1, N + 1, dtype=np.int64)
        M_vals = ks * sigma[1:] # sigma[1:] aligns with ks
        
        # Filter M <= 10^18
        # Since k<=10^7, sigma(k) <= ~1.5*10^7, M <= 1.5*10^14 < 10^18.
        # So no filtering needed for N up to 10^9.
        
        # Find most frequent M
        unique_M, inverse, counts = np.unique(M_vals, return_inverse=True, return_counts=True)
        
        max_count = np.max(counts)
        
        # Find indices of M with max count
        # There might be multiple M with the same max count.
        # We just need one set of k's.
        # Let's pick the first one.
        max_indices = np.where(counts == max_count)[0]
        
        if len(max_indices) == 0:
            return 0, []
            
        # Pick the first M with max count
        best_M_idx = max_indices[0]
        best_M = int(unique_M[best_M_idx])
        
        # Find all k's corresponding to this M
        # inverse maps each element in M_vals to its index in unique_M
        # So we want indices where inverse == best_M_idx
        k_indices = np.where(inverse == best_M_idx)[0]
        # k values are k_indices + 1
        best_ks = (k_indices + 1).tolist()
        
        return best_M, best_ks

    best_M = 0
    best_ks = []
    best_count = 0
    
    # Try N=100000
    N = 100000
    M, ks = run_search(N)
    if len(ks) > best_count:
        best_M, best_ks, best_count = M, ks, len(ks)
        
    # Try N=1000000
    if time.time() < deadline - 10:
        N = 1000000
        M, ks = run_search(N)
        if len(ks) > best_count:
            best_M, best_ks, best_count = M, ks, len(ks)
            
    # Try N=10000000
    if time.time() < deadline - 20:
        N = 10000000
        M, ks = run_search(N)
        if len(ks) > best_count:
            best_M, best_ks, best_count = M, ks, len(ks)
            
    # Try N=20000000
    if time.time() < deadline - 40:
        N = 20000000
        M, ks = run_search(N)
        if len(ks) > best_count:
            best_M, best_ks, best_count = M, ks, len(ks)

    if best_count == 0:
        return {'M': 333312, 'k': [336, 372, 434]}

    return {'M': int(best_M), 'k': [int(k) for k in best_ks]}
