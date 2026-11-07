import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Try progressively larger K values
    for K in [5_000_000, 10_000_000, 20_000_000, 30_000_000, 50_000_000]:
        if time.time() > deadline - 10.0:
            break
        
        try:
            sigma = np.zeros(K + 1, dtype=np.int64)
            for i in range(1, K + 1):
                sigma[i::i] += i
                if i % 5000000 == 0 and time.time() > deadline - 10.0:
                    break
            
            if time.time() > deadline - 5.0:
                continue
            
            # Compute M = k * sigma[k]
            ks = np.arange(1, K + 1, dtype=np.int64)
            Ms = ks * sigma[1:]
            
            # Filter M <= 10^18
            valid = Ms <= 10**18
            Ms = Ms[valid]
            ks = ks[valid]
            
            # Find most frequent M
            if len(Ms) == 0:
                continue
            
            unique_Ms, counts, inverse = np.unique(Ms, return_counts=True, return_inverse=True)
            max_count = counts.max()
            
            if max_count > 5:  # Better than current
                best_M = unique_Ms[counts == max_count][0]
                best_ks = ks[inverse == np.where(counts == max_count)[0][0]]
                return {'M': int(best_M), 'k': [int(x) for x in best_ks]}
                
        except Exception:
            continue
    
    # Fallback
    return {'M': 333312, 'k': [336, 372, 434]}
