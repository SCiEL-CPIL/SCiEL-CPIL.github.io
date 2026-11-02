import numpy as np
import time

D = 11

def construct():
    start_time = time.time()
    deadline = 110.0 # 10s margin for 120s limit
    
    vectors = []
    # Generate all vectors with exactly two non-zero entries, each being +/- 1
    # There are C(11, 2) = 55 pairs of indices.
    # For each pair, 4 sign combinations.
    # Total 220 vectors.
    
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    if time.time() > deadline:
                        break
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    if not vectors:
        # Fallback if something went wrong, though unlikely
        return np.array([], dtype=np.int64).reshape(0, D)
        
    return np.array(vectors, dtype=np.int64)
