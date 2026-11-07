import math
import time
import sys
import numpy as np
from collections import defaultdict

def _sigma(n):
    """Exact sigma(n) via trial division (with isqrt cutoff)."""
    if n <= 0:
        return 0
    s = 1
    x = n
    p = 2
    while p * p <= x:
        if x % p == 0:
            term = 1
            pk = 1
            while x % p == 0:
                x //= p
                pk *= p
                term += pk
            s *= term
        p += 1 if p == 2 else 2
    if x > 1:
        s *= (1 + x)
    return s

def construct():
    t0 = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill

    # Fallback: known 3-way collision
    best = {'M': 333312, 'k': [336, 372, 434]}
    best_len = len(best['k'])

    # Try progressively larger sieve sizes
    # Using numpy for speed
    sizes = [10**6, 5*10**6, 10**7, 2*10**7, 5*10**7]
    
    for N in sizes:
        if time.time() - t0 > DEADLINE:
            break
        try:
            # Initialize sigma array
            # We need int64 to hold sigma values and products
            sigma = np.zeros(N + 1, dtype=np.int64)
            
            # Sieve
            # This loop is O(N log N)
            for i in range(1, N + 1):
                sigma[i::i] += i
                
            # Calculate M = k * sigma(k)
            # We need to handle the product carefully
            # k is index, sigma is value
            # k is up to 5*10^7, sigma is up to ~ 10^8, product up to 5*10^15 < 10^18
            # So int64 is safe.
            
            # Create array of k values
            k_vals = np.arange(1, N + 1, dtype=np.int64)
            
            # Calculate M values
            M_vals = k_vals * sigma
            
            # Filter M <= 10^18
            # Also M > 0
            valid_mask = (M_vals <= 10**18) & (M_vals > 0)
            
            valid_M = M_vals[valid_mask]
            valid_k = k_vals[valid_mask]
            
            # Group by M
            # Sorting valid_M and then grouping is efficient
            # Or use a dictionary. With 10^7 elements, dict might be slow/memory heavy.
            # Sorting is O(N log N).
            
            # To save time, if N is large, sorting 10^7 items takes ~1s.
            # Let's try sorting.
            
            # Sort by M
            sort_idx = np.argsort(valid_M, kind='stable')
            sorted_M = valid_M[sort_idx]
            sorted_k = valid_k[sort_idx]
            
            # Find boundaries where M changes
            # diff > 0 indicates a new group
            diffs = np.diff(sorted_M)
            change_indices = np.where(diffs > 0)[0]
            # Add end index
            # The groups are [0, change_indices[0]], [change_indices[0]+1, change_indices[1]], ...
            
            # Calculate lengths of groups
            # Lengths are differences between consecutive change indices
            # We need to be careful with the start and end.
            
            # Append -1 and len to handle boundaries
            # indices: 0, c0, c1, ..., c_last, len
            # Length of group i is end_i - start_i
            # start_0 = 0, end_0 = c0
            # start_1 = c0+1, end_1 = c1
            # This logic is slightly tricky with numpy.
            
            # Alternative: use np.unique with return_counts
            # unique_M, counts, inverse_indices = np.unique(valid_M, return_counts=True, return_inverse=True)
            # This is implemented in C and fast.
            
            unique_M, counts, inverse_indices = np.unique(valid_M, return_counts=True, return_inverse=True)
            
            # Find max count
            if len(counts) > 0:
                max_count = np.max(counts)
                if max_count > best_len:
                    # Find which M has this count
                    # There might be multiple, pick the first one
                    best_M_idx = np.argmax(counts)
                    best_M = unique_M[best_M_idx]
                    
                    # Retrieve k values for this M
                    # valid_k[inverse_indices == best_M_idx]
                    # This creates a boolean mask of size N, which is okay.
                    mask = (inverse_indices == best_M_idx)
                    candidate_ks = valid_k[mask]
                    
                    # Convert to Python list of ints
                    # Verify with exact sigma
                    verified = []
                    for k in candidate_ks:
                        k_int = int(k)
                        if k_int * _sigma(k_int) == best_M and k_int <= math.isqrt(best_M):
                            verified.append(k_int)
                    
                    if len(verified) > best_len:
                        best_len = len(verified)
                        best = {'M': int(best_M), 'k': verified}
                        
        except Exception as e:
            # If numpy fails (memory error), break
            break

    # Ensure types
    M = int(best['M'])
    ks = [int(x) for x in best['k']]
    # Verify once more for safety
    final_ks = []
    for k in ks:
        if isinstance(k, int) and not isinstance(k, bool) and 1 <= k <= math.isqrt(M) and k * _sigma(k) == M:
            final_ks.append(k)
    # Deduplicate preserving order
    seen = set()
    dedup = []
    for k in final_ks:
        if k not in seen:
            seen.add(k)
            dedup.append(k)
    if len(dedup) == 0:
        # ultimate fallback
        return {'M': 333312, 'k': [336, 372, 434]}
    return {'M': M, 'k': dedup}
