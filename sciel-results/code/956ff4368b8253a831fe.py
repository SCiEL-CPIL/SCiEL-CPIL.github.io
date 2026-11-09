import numpy as np
from math import isqrt

def construct():
    N = 30000000
    # Compute sigma using NumPy for speed
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    # Compute M = k * sigma[k] for k = 1..N
    k_vals = np.arange(1, N + 1, dtype=np.int64)
    M_vals = k_vals * sigma[1:]
    
    # Filter M <= 10^18
    mask = M_vals <= 10**18
    M_vals = M_vals[mask]
    k_vals = k_vals[mask]
    
    if len(M_vals) == 0:
        return {'M': 1, 'k': [1]}
        
    # Find the M with the most occurrences
    unique_M, counts = np.unique(M_vals, return_counts=True)
    max_count = np.max(counts)
    # Find the first M with max_count
    idx_max = np.argmax(counts)
    best_M = unique_M[idx_max]
    
    # Get all k for this M
    k_list = k_vals[M_vals == best_M]
    
    # Truncate to 1000 if necessary
    if len(k_list) > 1000:
        k_list = k_list[:1000]
        
    # Convert to Python types
    best_M_int = int(best_M)
    k_list_py = [int(k) for k in k_list]
    
    return {'M': best_M_int, 'k': k_list_py}
