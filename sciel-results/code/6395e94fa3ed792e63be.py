"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Rank is 84
    rank = 84
    
    # Pre-allocate lists for U, V, W
    # Using lists of lists to ensure exact integer arithmetic
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Define the 6 Strassen blocks
    # Each block is (i_offset, j_offset, k_offset)
    # i is always 0..1 for these blocks
    # j is 0..1 or 2..3
    # k is 0..1, 2..3, or 4..5
    blocks = []
    for k_off in [0, 2, 4]:
        for j_off in [0, 2]:
            blocks.append((0, j_off, k_off))
            
    # Strassen coefficients for U, V, W
    # U is 2x2, V is 2x2, W is 2x2 (local)
    # We define the 7 rank-1 tensors
    strassen_terms = [
        # M1: (A00+A11)(B00+B11) -> C00+ C11+ C10- C01?
        # C00=1, C01=0, C10=-1, C11=1
        {'U': [(0,0,1), (1,1,1)], 'V': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,0,-1), (1,1,1)]},
        # M2: (A10+A11)B00 -> C10+ C11-
        {'U': [(1,0,1), (1,1,1)], 'V': [(0,0,1)], 'W': [(1,0,1), (1,1,-1)]},
        # M3: A00(B01-B11) -> C01+ C10+
        {'U': [(0,0,1)], 'V': [(0,1,1), (1,1,-1)], 'W': [(0,1,1), (1,0,1)]},
        # M4: A11(B10-B00) -> C00- C01+
        {'U': [(1,1,1)], 'V': [(1,0,1), (0,0,-1)], 'W': [(0,0,-1), (0,1,1)]},
        # M5: (A00+A01)B11 -> C00+ C11-
        {'U': [(0,0,1), (0,1,1)], 'V': [(1,1,1)], 'W': [(0,0,1), (1,1,-1)]},
        # M6: (A10-A00)(B00+B01) -> C10+ C11+
        {'U': [(1,0,1), (0,0,-1)], 'V': [(0,0,1), (0,1,1)], 'W': [(1,0,1), (1,1,1)]},
        # M7: (A01-A11)(B10+B11) -> C00- C01+
        {'U': [(0,1,1), (1,1,-1)], 'V': [(1,0,1), (1,1,1)], 'W': [(0,0,-1), (0,1,1)]}
    ]
    
    # Track which global (i,j,k) are covered by Strassen
    covered = set()
    
    # Add Strassen terms
    for i_off, j_off, k_off in blocks:
        for term in strassen_terms:
            # U part
            for (li, lj, val) in term['U']:
                i_glob = i_off + li
                j_glob = j_off + lj
                U[i_glob * m + j_glob][col_idx] += val
                covered.add((i_glob, j_glob, k_off + 0)) # Just to mark area? No, need precise marking later
                covered.add((i_glob, j_glob, k_off + 1))
            
            # V part
            for (lj, lk, val) in term['V']:
                j_glob = j_off + lj
                k_glob = k_off + lk
                V[j_glob * p + k_glob][col_idx] += val
            
            # W part
            for (li, lk, val) in term['W']:
                i_glob = i_off + li
                k_glob = k_off + lk
                W[k_glob * n + i_glob][col_idx] += val
                
            col_idx += 1
    
    # Add Schoolbook terms for uncovered indices
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this (i,j,k) is part of a Strassen block
                # Strassen blocks are: i in {0,1}, j in {0,1,2,3}
                if i < 2 and j < 4:
                    continue
                
                U[i * m + j][col_idx] = 1
                V[j * p + k][col_idx] = 1
                W[k * n + i][col_idx] = 1
                col_idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
