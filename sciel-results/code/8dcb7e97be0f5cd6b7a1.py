import math
import numpy as np

def construct():
    # New Approach: Vectorized Sieve + np.unique counting
    # This leverages NumPy's C-implemented sort/unique for grouping, which is much faster
    # than Python-level dictionary grouping for large N.
    
    # Choose N based on memory. 
    # sigma: int32 (4 bytes)
    # M_arr: int64 (8 bytes)
    # unique/counts: ~16 bytes per element worst case
    # Total ~ 28 bytes per element.
    # 30_000_000 * 28 = 840 MB. Should be safe on most 2GB+ systems.
    N = 30_000_000
    
    # 1. Compute sigma using sieve
    sigma = np.zeros(N + 1, dtype=np.int32)
    
    # Sieve loop
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    # 2. Compute M = k * sigma(k)
    k_vals = np.arange(1, N + 1, dtype=np.int64)
    M_vals = k_vals * sigma
    
    # Filter M <= 10^18
    mask = M_vals <= 10**18
    M_vals = M_vals[mask]
    k_vals = k_vals[mask]
    
    if len(M_vals) == 0:
        return {'M': 1, 'k': [1]}
        
    # 3. Find most frequent M
    unique_M, counts = np.unique(M_vals, return_counts=True)
    
    max_idx = np.argmax(counts)
    best_M = int(unique_M[max_idx])
    
    # 4. Retrieve k's for best_M
    target_ks = k_vals[M_vals == best_M]
    
    # Convert to Python list of ints
    return {'M': best_M, 'k': [int(k) for k in target_ks]}
