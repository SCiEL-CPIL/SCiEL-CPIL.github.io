"""Matrix multiplication 2,4,8 tensor decomposition"""
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # A is n x m, B is m x p, C is n x p
    
    # Rank 56 using Strassen for 2x2x2 blocks
    # We split the j dimension (size 4) into 2 blocks of size 2
    # and the k dimension (size 8) into 4 blocks of size 2
    # Each block uses 7 Strassen multiplications
    # Total rank = 2 * 4 * 7 = 56
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1: (a00+a11)(b00+b11)
    # M2: (a10+a11)b00
    # M3: a00(b01-b11)
    # M4: a11(b10-b00)
    # M5: (a00+a01)b11
    # M6: (a10-a00)(b00+b01)
    # M7: (a01-a11)(b10+b11)
    
    # u: (i, j) -> coeff for A[i,j]
    # v: (j, k) -> coeff for B[j,k]
    # w: (k, i) -> coeff for C[i,k] (but W index is k*n+i)
    
    strassen_u = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1}
    ]
    
    strassen_v = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1}
    ]
    
    strassen_w = [
        {(0,0): 1, (1,1): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(0,0): -1, (1,0): 1},
        {(1,1): 1},
        {(0,0): 1}
    ]
    
    for s in range(2):  # j split: 0 for j in {0,1}, 1 for j in {2,3}
        for t in range(4):  # k split: 0..3 for k in {0,1}, {2,3}, {4,5}, {6,7}
            for r in range(7):
                # Update U
                for (i_loc, j_loc), val in strassen_u[r].items():
                    i_glob = i_loc
                    j_glob = 2 * s + j_loc
                    U[i_glob * m + j_glob, col] += val
                
                # Update V
                for (j_loc, k_loc), val in strassen_v[r].items():
                    j_glob = 2 * s + j_loc
                    k_glob = 2 * t + k_loc
                    V[j_glob * p + k_glob, col] += val
                
                # Update W
                for (k_loc, i_loc), val in strassen_w[r].items():
                    k_glob = 2 * t + k_loc
                    i_glob = i_loc
                    W[k_glob * n + i_glob, col] += val
                
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
