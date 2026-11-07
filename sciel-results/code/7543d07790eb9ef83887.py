"""Search for M with many distinct k where k*sigma(k)=M, k<=isqrt(M)."""
import math
import time
import sys
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    # Try multiple K values, starting large and reducing if needed
    K_values = [20000000, 15000000, 10000000, 5000000, 1000000]
    
    for K in K_values:
        if time.time() > deadline:
            break
        try:
            # 1. Compute Sigma using NumPy
            sigma = np.zeros(K + 1, dtype=np.int64)
            for d in range(1, K + 1):
                sigma[d::d] += d
            
            # 2. Group by M using vectorized operations where possible
            # Convert to Python list for grouping
            sigma_list = sigma.tolist()
            
            groups = defaultdict(list)
            g = groups
            for k in range(1, K + 1):
                M = k * sigma_list[k]
                if M > 10**18:
                    continue
                g[M].append(k)
                
            # 3. Find best
            for M, ks in g.items():
                cnt = len(ks)
                if cnt > best_count:
                    best_count = cnt
                    best_M = M
                    best_k = ks
            
            # If we found a good result, stop early
            if best_count >= 8:
                break
                
        except MemoryError:
            continue
        except Exception:
            continue

    # Final safety: ensure types are plain Python int and list
    best_k = [int(x) for x in best_k]
    best_M = int(best_M)
    
    return {'M': best_M, 'k': best_k}
