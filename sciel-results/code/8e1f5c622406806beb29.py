"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 120.0 - 5.0 # 5s safety margin

    # Generate E8 roots (240 vectors in Z^8)
    e8_roots = []
    # Type 1: vectors with exactly two non-zero entries, each +-1
    for i, j in itertools.combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * 8
                v[i] = si
                v[j] = sj
                e8_roots.append(v)
                
    # Type 2: vectors with exactly eight non-zero entries, each +-1, even number of -1s
    for bits in itertools.product([1, -1], repeat=8):
        if bits.count(-1) % 2 == 0:
            e8_roots.append(list(bits))
            
    # Generate D3 roots (12 vectors in Z^3)
    # D3 is isomorphic to A3, roots are e_i - e_j
    # In Z^3 representation: two +-1 and one 0
    d3_roots = []
    for i, j in itertools.combinations(range(3), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * 3
                v[i] = si
                v[j] = sj
                d3_roots.append(v)
                
    # Construct the full set of vectors in R^11
    # v = (e, d) where e in E8, d in D3
    # Norm squared of e is 2, norm squared of d is 2. Total norm squared is 4.
    # Max dot product is 1 (from E8) + 1 (from D3) = 2.
    # Condition: ||v_i - v_j||^2 >= 4 => 8 - 2 v_i.v_j >= 4 => v_i.v_j <= 2.
    # This is satisfied.
    
    vectors = []
    for e in e8_roots:
        for d in d3_roots:
            # e has 8 elements, d has 3 elements. Total 11.
            v = e + d
            vectors.append(v)
            
            # Time check
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
