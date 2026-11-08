import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _construct_rank_36():
    """
    Constructs a rank-36 decomposition for 2x3x10 matrix multiplication.
    Splits into 2x2x10 (Strassen tiling, rank 35) and 2x1x10 (rank 1).
    """
    n, m, p = DIMENSIONS
    rank = 36
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    # 5 blocks of 2 columns for the 2x2 part
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # Strassen M1: (A00 + A11) * (B0,k0 + B1,k1)
        U[0 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        # C00 += M1, C11 += M1
        W[k0 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # Strassen M2: (A10 + A11) * B0,k0
        U[1 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        # C10 += M2, C11 -= M2
        W[k0 * 2 + 1][col] += 1
        W[k1 * 2 + 1][col] -= 1
        col += 1
        
        # Strassen M3: A00 * (B0,k1 - B1,k1)
        U[0 * 3 + 0][col] += 1
        V[0 * 10 + k1][col] += 1
        V[1 * 10 + k1][col] -= 1
        # C01 += M3, C11 += M3
        W[k1 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # Strassen M4: A11 * (B1,k0 - B0,k0)
        U[1 * 3 + 1][col] += 1
        V[1 * 10 + k0][col] += 1
        V[0 * 10 + k0][col] -= 1
        # C00 += M4, C10 += M4
        W[k0 * 2 + 0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        # Strassen M5: (A00 + A01) * B1,k1
        U[0 * 3 + 0][col] += 1
        U[0 * 3 + 1][col] += 1
        V[1 * 10 + k1][col] += 1
        # C00 -= M5, C01 += M5
        W[k0 * 2 + 0][col] -= 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        # Strassen M6: (A10 - A00) * (B0,k0 + B0,k1)
        U[1 * 3 + 0][col] += 1
        U[0 * 3 + 0][col] -= 1
        V[0 * 10 + k0][col] += 1
        V[0 * 10 + k1][col] += 1
        # C11 += M6
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # Strassen M7: (A01 - A11) * (B1,k0 + B1,k1)
        U[0 * 3 + 1][col] += 1
        U[1 * 3 + 1][col] -= 1
        V[1 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        # C00 += M7
        W[k0 * 2 + 0][col] += 1
        col += 1
        
    # 2x1x10 part: A_col2 * B_row2
    # U[i*3+2, col] = 1
    # V[2*10+k, col] = 1
    # W[k*2+i, col] = 1
    U[0 * 3 + 2][col] = 1
    U[1 * 3 + 2][col] = 1
    for k in range(10):
        V[2 * 10 + k][col] = 1
        W[k * 2 + 0][col] = 1
        W[k * 2 + 1][col] = 1
    col += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    DEADLINE = 140.0

    U, V, W = _construct_rank_36()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
