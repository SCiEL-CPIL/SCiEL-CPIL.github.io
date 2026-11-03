"""Matrix multiplication tensor decomposition for 3x3 times 3x6."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    # Decompose into two 3x3x3 problems
    # Rank of 3x3x3 schoolbook is 27
    # Total rank = 2 * 27 = 54
    
    rank_slice = 27
    rank = 2 * rank_slice
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Generate schoolbook factors for 3x3x3
    U_slice = np.zeros((9, rank_slice))
    V_slice = np.zeros((9, rank_slice))
    W_slice = np.zeros((9, rank_slice))
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_slice[i * 3 + j, col] = 1.0
                V_slice[j * 3 + k, col] = 1.0
                W_slice[k * 3 + i, col] = 1.0
                col += 1
                
    # Fill U: Same for both slices
    U[:, 0:rank_slice] = U_slice
    U[:, rank_slice:2*rank_slice] = U_slice
    
    # Fill V: First slice uses B1 (cols 0-2), Second slice uses B2 (cols 3-5)
    # V rows: j*6 + k
    for j in range(3):
        for k in range(3):
            # Slice 1: B1[j,k] corresponds to V[j*6 + k, 0:rank_slice]
            V[j * 6 + k, 0:rank_slice] = V_slice[j * 3 + k, 0:rank_slice]
            # Slice 2: B2[j,k] corresponds to V[j*6 + (k+3), rank_slice:2*rank_slice]
            V[j * 6 + (k + 3), rank_slice:2*rank_slice] = V_slice[j * 3 + k, 0:rank_slice]
            
    # Fill W: First slice uses C1 (cols 0-2), Second slice uses C2 (cols 3-5)
    # W rows: k*3 + i (convention W[k*n+i, r])
    for k in range(3):
        for i in range(3):
            # Slice 1: C1[i,k] corresponds to W[k*3 + i, 0:rank_slice]
            W[k * 3 + i, 0:rank_slice] = W_slice[k * 3 + i, 0:rank_slice]
            # Slice 2: C2[i,k] corresponds to W[(k+3)*3 + i, rank_slice:2*rank_slice]
            W[(k + 3) * 3 + i, rank_slice:2*rank_slice] = W_slice[k * 3 + i, 0:rank_slice]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
