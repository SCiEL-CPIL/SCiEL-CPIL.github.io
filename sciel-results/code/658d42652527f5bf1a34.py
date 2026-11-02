"""Matrix multiplication 2x4 x 4x4 tensor decomposition - attempting rank 27."""

import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    # Build rank-28 Strassen decomposition
    def build_strassen_28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]  # 8 x 28
        V = [[0] * rank for _ in range(m * p)]  # 16 x 28
        W = [[0] * rank for _ in range(p * n)]  # 8 x 28
        
        col = 0
        for b in range(2):  # B column block: 0 or 1
            for a_blk in range(2):  # A column block / B row block: 0 or 1
                for s in range(7):
                    c = col
                    j0 = 2 * a_blk
                    j1 = 2 * a_blk + 1
                    
                    # U side: A[i, 2*a_blk + j]
                    # U flat index: i * 4 + (2*a_blk + j)
                    U_contrib = {
                        0: [(0, j0, 1), (1, j1, 1)],
                        1: [(1, j0, 1), (1, j1, 1)],
                        2: [(0, j0, 1)],
                        3: [(1, j1, 1)],
                        4: [(0, j0, 1), (0, j1, 1)],
                        5: [(1, j0, 1), (0, j0, -1)],
                        6: [(0, j1, 1), (1, j1, -1)],
                    }
                    for (i, j, coeff) in U_contrib[s]:
                        U[i * 4 + j][c] += coeff
                    
                    # V side: B[2*a_blk + j, 2*b + k]
                    # V flat index: (2*a_blk + j) * 4 + (2*b + k)
                    V_contrib = {
                        0: [(2*a_blk, 2*b, 1), (2*a_blk+1, 2*b+1, 1)],
                        1: [(2*a_blk, 2*b, 1)],
                        2: [(2*a_blk, 2*b+1, 1), (2*a_blk+1, 2*b+1, -1)],
                        3: [(2*a_blk+1, 2*b, 1), (2*a_blk, 2*b, -1)],
                        4: [(2*a_blk+1, 2*b+1, 1)],
                        5: [(2*a_blk, 2*b, 1), (2*a_blk, 2*b+1, 1)],
                        6: [(2*a_blk+1, 2*b, 1), (2*a_blk+1, 2*b+1, 1)],
                    }
                    for (j, k, coeff) in V_contrib[s]:
                        V[j * 4 + k][c] += coeff
                    
                    # W side: C[i, 2*b + k]
                    # W flat index: (2*b + k) * 2 + i
                    W_contrib = {
                        0: {(0, 0): 1, (1, 1): 1},
                        1: {(1, 0): 1, (1, 1): -1},
                        2: {(0, 1): 1, (1, 1): 1},
                        3: {(0, 0): 1, (1, 0): 1},
                        4: {(0, 0): -1, (0, 1): 1},
                        5: {(1, 1): 1},
                        6: {(0, 0): 1},
                    }
                    for (i_loc, k_loc), coeff in W_contrib[s].items():
                        W[(2 * b + k_loc) * 2 + i_loc][c] += coeff
                    
                    col += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Validate the solution to be safe
    def validate(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    result = build_strassen_28()
    
    # Quick validation
    if validate(result["U"], result["V"], result["W"], 28):
        return result
    else:
        # Fallback to naive if validation fails (should not happen)
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i*m*p + j*p + k
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
