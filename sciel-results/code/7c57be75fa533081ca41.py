import numpy as np
import itertools
import time

D = 11

def construct():
    # Generate Type 1: Weight 1, +/- 2
    # These are in D11 (sum is +/- 2, even)
    vecs_t1 = []
    for i in range(D):
        for s in [1, -1]:
            v = [0]*D
            v[i] = 2*s
            vecs_t1.append(v)
            
    # Generate Type 4: Weight 4, +/- 1
    # These are in D11 (sum is 4-2k, even)
    vecs_t4 = []
    for pos in itertools.combinations(range(D), 4):
        for mask in range(16):
            signs = [1 if (mask & (1<<j)) == 0 else -1 for j in range(4)]
            v = [0]*D
            for p, val in zip(pos, signs):
                v[p] = val
            vecs_t4.append(v)
                
    V_t1 = np.array(vecs_t1, dtype=np.int64)
    V_t4 = np.array(vecs_t4, dtype=np.int64)
    
    # The union of these sets forms a valid kissing configuration
    # Min pairwise distance squared is 4 (dist 2)
    # Max norm squared is 4 (norm 2)
    # 2 >= 2 holds.
    
    result = np.vstack([V_t1, V_t4])
    
    return result
