import numpy as np
import itertools
import time

D = 11

def construct():
    # Generate all integer vectors with squared norm 2
    # These are of the form +/- e_i +/- e_j for i != j
    vectors = []
    
    # We track elapsed time to ensure we return within 120s
    start_time = time.time()
    deadline = 150.0  # Safety margin
    
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                vectors.append(v)
                
                # Check time
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    if not vectors:
        return np.array([]).reshape(0, D)
        
    return np.array(vectors, dtype=np.int64)
