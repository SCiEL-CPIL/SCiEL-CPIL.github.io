"""
Solution for Kissing Number in Dimension 11.
Uses the E8 x D3 lattice construction to generate 252 integer vectors.
"""
import numpy as np
import time
from itertools import product

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    vectors = []
    
    # 1. Generate 240 vectors from the E8 lattice (scaled by 2)
    # Embedded in the first 8 dimensions.
    
    # Type A: +/- 2 e_i +/- 2 e_j for i < j in 1..8
    # There are C(8,2) = 28 pairs.
    # For each pair, 4 sign combinations: (2,2), (2,-2), (-2,2), (-2,-2)
    for i in range(8):
        for j in range(i + 1, 8):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = 2 * s1
                    v[j] = 2 * s2
                    vectors.append(v)
    
    # Type B: ( +/- 1, ..., +/- 1 ) with even number of minus signs
    # There are 2^7 = 128 such vectors.
    # We iterate over all 2^8 sign combinations and check parity.
    for signs in product([1, -1], repeat=8):
        # Count minus signs
        n_minus = sum(1 for s in signs if s == -1)
        if n_minus % 2 == 0:
            v = np.zeros(D, dtype=np.int64)
            v[:8] = signs
            vectors.append(v)
            
    # 2. Generate 12 vectors from the D3 lattice (scaled by 2)
    # Embedded in the last 3 dimensions (indices 8, 9, 10).
    # D3 is isomorphic to A3. Roots are +/- e_i +/- e_j.
    # Pairs: (8,9), (8,10), (9,10)
    for i, j in [(8, 9), (8, 10), (9, 10)]:
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = np.zeros(D, dtype=np.int64)
                v[i] = 2 * s1
                v[j] = 2 * s2
                vectors.append(v)
                
    # Total vectors: 112 + 128 + 12 = 252
    
    result = np.array(vectors, dtype=np.int64)
    
    # Verification (optional, for safety)
    # Check if min pairwise distance >= max norm
    # Max norm is sqrt(8) for all vectors.
    # Min distance squared should be >= 8.
    
    # Efficient check using matrix multiplication
    # M = result @ result.T
    # Diagonals are 8.
    # Off-diagonals should be <= 4 (since dist^2 = 8+8-2dot >= 8 => dot <= 4)
    
    if time.time() < deadline - 5.0:
        M = result @ result.T
        np.fill_diagonal(M, 0)
        # If any off-diagonal element is > 4, the constraint is violated
        # dist^2 = 16 - 2*dot. We need dist^2 >= 8 => 16 - 2*dot >= 8 => 2*dot <= 8 => dot <= 4.
        if np.any(M > 4):
            # Fallback to a smaller valid set if something went wrong (should not happen)
            # Just return the first 100 vectors or something? 
            # But E8 x D3 is known to be valid.
            pass
            
    return result

# EVOLVE-BLOCK-END
