import time
import math
import numpy as np

def construct():
    deadline = time.time() + 140
    
    # Known best from seed/small search
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    def search(N):
        nonlocal best_M, best_k, best_count
        if time.time() > deadline - 15:
            return
        
        try:
            # Sieve for sigma(k)
            # Using numpy for speed
            sigma = np.zeros(N + 1, dtype=np.int64)
            
            # Optimization: Use a simple loop for small N? No, NumPy is faster for N > 10k.
            for d in range(1, N + 1):
                sigma[d::d] += d
                
            # Compute M = k * sigma(k)
            ks = np.arange(1, N + 1, dtype=np.int64)
            Ms = ks * sigma[1:]
            
            # We only care about M <= 10^18
            # Max k=2e7, sigma(k) approx k log log k ~ 2e7 * 4 = 8e7.
            # M approx 1.6e15. Well within int64 and 10^18.
            
            # Find the M with the highest frequency
            # np.unique is efficient for large arrays
            unique_Ms, counts = np.unique(Ms, return_counts=True)
            max_idx = np.argmax(counts)
            count = int(counts[max_idx])
            M_val = int(unique_Ms[max_idx])
            
            if count > best_count:
                # Retrieve the k values for this M
                # Use boolean indexing
                mask = Ms == unique_Ms[max_idx]
                ks_found = ks[mask]
                
                # Convert to list of Python ints
                k_list = [int(x) for x in ks_found]
                
                # Verify constraints: k <= isqrt(M)
                # As discussed, k < sqrt(M) for k > 1.
                # k=1 implies M=1.
                isqrt_M = math.isqrt(M_val)
                k_list = [k for k in k_list if k <= isqrt_M]
                
                # Ensure distinctness (they are from unique ks array, so distinct)
                if len(k_list) == count:
                    best_M = M_val
                    best_k = k_list
                    best_count = count
        except Exception:
            # If memory error or other issue, skip this N
            pass

    # Try increasing limits
    # 5e6 is fast (~2s)
    # 1e7 is moderate (~5s)
    # 2e7 is slower (~15s)
    # 5e7 might be too slow/memory heavy for Python loop overhead in sieve
    
    search(5_000_000)
    if time.time() < deadline - 30:
        search(10_000_000)
    if time.time() < deadline - 60:
        search(20_000_000)
        
    # Final cleanup
    isqrt_M = math.isqrt(best_M)
    best_k = [k for k in best_k if k <= isqrt_M]
    
    return {'M': int(best_M), 'k': [int(x) for x in best_k[:1000]]}
