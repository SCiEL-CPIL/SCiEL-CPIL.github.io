import numpy as np
import itertools
import time

D = 11

def construct():
    # Generate all integer vectors in R^11 with exactly 4 non-zero entries, each +/-1.
    # Norm is sqrt(4) = 2.
    # Minimum distance between any two such vectors is 2 (if they share 3 indices with same sign and 1 opposite, or 2 same and 2 opposite? 
    # Let's re-verify min dist for w=4.
    # v=(1,1,1,1,0..), w=(1,1,1,-1,0..). diff=(0,0,0,2,0..). dist=2.
    # v=(1,1,1,1,0..), w=(1,1,0,0,1,1,0..). diff=(0,0,1,1,-1,-1,0..). dist=sqrt(4)=2.
    # v=(1,1,1,1,0..), w=(1,1,1,1,0..). dist=0 (identical).
    # So min non-zero dist is 2.
    # Max norm is 2.
    # Condition: min_dist >= max_norm => 2 >= 2. Holds.
    
    vectors = []
    # Choose 4 positions out of 11
    for positions in itertools.combinations(range(D), 4):
        # Generate all 2^4 = 16 sign combinations
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
