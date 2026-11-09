import time
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 115.0 # 5s margin
    
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    # Try larger limits with NumPy acceleration
    limits = [10_000_000, 30_000_000, 50_000_000]
    
    for N in limits:
        # Check time
        if time.time() - start_time > deadline - 15.0:
            break
            
        try:
            # NumPy sieve
            sigma = np.zeros(N + 1, dtype=np.int64)
            for i in range(1, N + 1):
                sigma[i::i] += i
            
            # Convert to list for faster indexing? Or keep as array?
            # Array indexing is fast.
            
            # Check time
            if time.time() - start_time > deadline - 5.0:
                break

            counts = defaultdict(list)
            
            # Iterate
            # Using a Python loop over N is still O(N).
            # N=50M is 50 million iterations. In Python, this is ~5-10s.
            # Total time: Sieve (NumPy, fast) + Loop (Python, ~10s).
            
            for k in range(1, N + 1):
                s_k = sigma[k]
                M_val = k * s_k
                
                if M_val <= 10**18:
                    counts[M_val].append(k)
                    if len(counts[M_val]) > best_count:
                        best_count = len(counts[M_val])
                        best_M = M_val
                        best_k = counts[M_val][:]
                        
        except MemoryError:
            break
        except Exception:
            break

    return {'M': int(best_M), 'k': best_k}
