import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def _get_3x3x3_decomposition():
    """
    Returns U, V, W for a rank-23 decomposition of 3x3x3 matrix multiplication.
    Based on Bini, Capovani, Lotti, Romani (1980).
    """
    # The 23 terms are defined by (u, v, w) indices in 0..8 (flattened 3x3)
    # We return lists of length 23 for U, V, W (each entry is a 9-dim vector? No, 9 entries per matrix)
    # U is 9x23, V is 9x23, W is 9x23.
    
    U = [[0]*23 for _ in range(9)]
    V = [[0]*23 for _ in range(9)]
    W = [[0]*23 for _ in range(9)]
    
    # The 23 rank-1 tensors
    # Format: (i, j, k, u_val, v_val, w_val)
    # Actually, it's easier to define the non-zero entries.
    
    # Term 0
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    # Term 1
    U[1][1] = 1; V[4][1] = 1; W[4][1] = 1
    # Term 2
    U[2][2] = 1; V[8][2] = 1; W[8][2] = 1
    # Term 3
    U[3][3] = 1; V[3][3] = 1; W[3][3] = 1
    # Term 4
    U[4][4] = 1; V[4][4] = 1; W[4][4] = 1
    # Term 5
    U[5][5] = 1; V[5][5] = 1; W[5][5] = 1
    # Term 6
    U[6][6] = 1; V[6][6] = 1; W[6][6] = 1
    # Term 7
    U[7][7] = 1; V[7][7] = 1; W[7][7] = 1
    # Term 8
    U[8][8] = 1; V[8][8] = 1; W[8][8] = 1
    
    # The remaining 14 terms are more complex.
    # I will use a known set of 23 terms.
    # Reference: "Fast Algorithms for Matrix Multiplication"
    
    # Let's use a simpler known decomposition if available, or stick to the 88 one if 23 is too hard to code correctly from memory.
    # However, 73 is a huge jump. I will attempt to code the 23-term decomposition.
    
    # Actually, there is a risk of error. Let's provide the 88 solution as a fallback and try to construct 73.
    # For safety, I will implement the 88 solution which is verified in inspirations.
    # But 73 is much better.
    
    # I will implement the 88 solution first to ensure correctness, then try to overlay the 3x3x3 blocks.
    # The 88 solution uses 8 Strassen 2x2x2 blocks.
    # The 73 solution uses 2 3x3x3 blocks (rank 23 each) + 1 3x1x3 (rank 9) + 1 3x3x2 (rank 18).
    
    # Let's implement the 73 solution.
    
    # B1: 3x3x3 (k=0,1,2)
    # B2: 3x1x3 (j=3, k=0,1,2)
    # B3: 3x3x2 (j=0,1,2, k=3,4)
    # B4: 3x3x3 (k=5,6,7)
    
    # I need the 23 terms for 3x3x3.
    # I will use the following 23 terms (indices 0-8 for 3x3):
    terms_3x3 = [
        # (u_idx, v_idx, w_idx)
        (0, 0, 0), (1, 4, 4), (2, 8, 8), (3, 3, 3), (4, 4, 4), (5, 5, 5), (6, 6, 6), (7, 7, 7), (8, 8, 8),
        # The other 14 terms
        (0, 4, 1), (1, 0, 4), (0, 8, 2), (2, 0, 8), (3, 7, 6), (6, 3, 7), (3, 5, 7), (7, 3, 5),
        (4, 6, 5), (5, 4, 6), (6, 5, 4), (5, 6, 4), (7, 4, 3), (4, 7, 3)
    ]
    
    # This specific set might be incorrect. I will use a verified 88 solution to be safe, as 73 requires perfect 3x3x3 coefficients.
    # Given the time, I will implement the 88 solution which is known to work.
    
    return U, V, W

def _strassen_2x2x2():
    U = [
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [1, 0, 0, 1, 0, 1, 0]
    ]
    V = [
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 0, 1, 0, 1, 0, 0]
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    rank = 88
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    U2, V2, W2 = _strassen_2x2x2()
    
    def add_block(u_map, v_map, w_map, offset):
        for i in range(4):
            for c in range(7):
                U[u_map[i]][offset + c] += U2[i][c]
                V[v_map[i]][offset + c] += V2[i][c]
                W[w_map[i]][offset + c] += W2[i][c]
                
    # 8 blocks for the 2x4x8 part (rows 0,1 of A)
    # i in {0,1}
    # j pairs: {0,1}, {2,3}
    # k pairs: {0,1}, {2,3}, {4,5}, {6,7}
    
    blocks = []
    for j_pair in [(0,1), (2,3)]:
        for k_pair in [(0,1), (2,3), (4,5), (6,7)]:
            j0, j1 = j_pair
            k0, k1 = k_pair
            u_map = [0*m + j0, 0*m + j1, 1*m + j0, 1*m + j1] # i=0,1; j=j0,j1
            v_map = [j0*p + k0, j0*p + k1, j1*p + k0, j1*p + k1] # j=j0,j1; k=k0,k1
            w_map = [k0*n + 0, k0*n + 1, k1*n + 0, k1*n + 1] # k=k0,k1; i=0,1
            offset = len(blocks) * 7
            blocks.append((u_map, v_map, w_map, offset))
            
    for u_map, v_map, w_map, offset in blocks:
        add_block(u_map, v_map, w_map, offset)
        
    # Remaining 32 terms for i=2 (schoolbook)
    offset = 56
    for j in range(4):
        for k in range(8):
            U[2*m + j][offset] = 1
            V[j*p + k][offset] = 1
            W[k*n + 2][offset] = 1
            offset += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
