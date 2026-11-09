import time
import math
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115.0  # Safety margin before 120s limit
    
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    # Try increasing N values
    N_values = [5_000_000, 10_000_000, 20_000_000, 50_000_000, 100_000_000]
    
    for N in N_values:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5.0:
            break
        
        # Estimate time for NumPy sieve
        est_time = 0.0000003 * N + 0.00000001 * N * (math.log(N) + 1)
        if elapsed + est_time > DEADLINE - 3.0:
            continue
        
        try:
            sigma = np.zeros(N + 1, dtype=np.int64)
            
            # Sieve using NumPy slicing with periodic time checks
            aborted = False
            for i in range(1, N + 1):
                if i % 2000000 == 0:
                    if time.time() - start_time > DEADLINE - 15.0:
                        aborted = True
                        break
                sigma[i::i] += i
            
            if aborted:
                continue
            
            # Build groups: M -> list of k
            groups = defaultdict(list)
            for k in range(1, N + 1):
                M_val = int(k * sigma[k])
                if M_val <= 10**18:
                    groups[M_val].append(k)
            
            # Find best M (k <= sqrt(M) is automatically satisfied)
            for M_val, ks in groups.items():
                if len(ks) > best_count:
                    best_count = len(ks)
                    best_M = M_val
                    best_k = ks
                        
        except Exception:
            continue
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10.0:
            break
    
    # Ensure types are correct
    best_k = [int(x) for x in best_k]
    best_M = int(best_M)
    
    return {'M': best_M, 'k': best_k}
