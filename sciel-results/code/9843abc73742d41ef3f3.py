import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    candidates = []
    
    # Generate D8 roots in dimensions 0-7
    # Vectors of form (+-1, +-1, 0, ..., 0) with two non-zeros
    for i in range(8):
        for j in range(i + 1, 8):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * D
                    v[i] = s1
                    v[j] = s2
                    candidates.append(v)
                    
    # Generate D3 roots in dimensions 8-10
    # Vectors of form (0, ..., 0, +-1, +-1, 0) with two non-zeros in last 3 dims
    for i in range(8, 11):
        for j in range(i + 1, 11):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * D
                    v[i] = s1
                    v[j] = s2
                    candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    
    # Verify validity just in case
    # Norm sq should be 2 for all
    norms_sq = np.sum(candidates ** 2, axis=1)
    if not np.all(norms_sq == 2):
        # Fallback to D11 if something went wrong
        candidates = []
        for i in range(D):
            for j in range(i + 1, D):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        v = [0] * D
                        v[i] = s1
                        v[j] = s2
                        candidates.append(v)
        candidates = np.array(candidates, dtype=np.int64)
        
    return candidates
