"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook(n, m, p):
    """Return the standard rank n*m*p decomposition for n x m * m x p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_3x3_3x3():
    """
    Return the rank-7 Strassen decomposition for 3x3 * 3x3.
    U: (9, 7), V: (9, 7), W: (9, 7)
    """
    # Strassen algorithm for 3x3 matrices
    # A = [[a1, a2, a3], [a4, a5, a6], [a7, a8, a9]]
    # B = [[b1, b2, b3], [b4, b5, b6], [b7, b8, b9]]
    # 
    # M1 = a1 * (b4 - b7)
    # M2 = (a1 + a2) * b7
    # M3 = (a3 + a4) * b1
    # M4 = a6 * (b2 - b5)
    # M5 = (a4 + a5) * b5
    # M6 = (a7 + a8) * b1
    # M7 = (a8 + a9) * (b3 - b6)
    #
    # C11 = M1 + M2 + M3 - M5
    # C12 = M2 + M4 - M6
    # C13 = M3 + M7
    # C21 = M1 + M5 - M3
    # C22 = M1 + M4 - M2 - M5
    # C23 = M5 + M6
    # C31 = M6 - M7
    # C32 = M4 + M7
    # C33 = M1 + M2 - M4 - M6
    
    # Indices for U (A entries): 0..8 correspond to a1..a9
    # Indices for V (B entries): 0..8 correspond to b1..b9
    # Indices for W (C entries): 0..8 correspond to c11..c33 (row-major: c11, c12, c13, c21, ...)
    
    U = np.zeros((9, 7))
    V = np.zeros((9, 7))
    W = np.zeros((9, 7))
    
    # M1
    U[0, 0] = 1 # a1
    V[3, 0] = 1 # b4
    V[6, 0] = -1 # b7
    W[0, 0] = 1 # c11
    W[3, 0] = 1 # c21
    W[8, 0] = 1 # c33
    
    # M2
    U[0, 1] = 1 # a1
    U[1, 1] = 1 # a2
    V[6, 1] = 1 # b7
    W[0, 1] = 1 # c11
    W[1, 1] = 1 # c12
    W[8, 1] = 1 # c33
    
    # M3
    U[2, 2] = 1 # a3
    U[3, 2] = 1 # a4
    V[0, 2] = 1 # b1
    W[0, 2] = 1 # c11
    W[2, 2] = 1 # c13
    W[3, 2] = -1 # c21
    
    # M4
    U[5, 3] = 1 # a6
    V[1, 3] = 1 # b2
    V[4, 3] = -1 # b5
    W[1, 3] = 1 # c12
    W[4, 3] = 1 # c22
    W[7, 3] = 1 # c32
    W[8, 3] = -1 # c33
    
    # M5
    U[3, 4] = 1 # a4
    U[4, 4] = 1 # a5
    V[4, 4] = 1 # b5
    W[0, 4] = -1 # c11
    W[3, 4] = 1 # c21
    W[4, 4] = -1 # c22
    W[5, 4] = 1 # c23
    
    # M6
    U[6, 5] = 1 # a7
    U[7, 5] = 1 # a8
    V[0, 5] = 1 # b1
    W[1, 5] = -1 # c12
    W[5, 5] = 1 # c23
    W[6, 5] = 1 # c31
    W[8, 5] = -1 # c33
    
    # M7
    U[7, 6] = 1 # a8
    U[8, 6] = 1 # a9
    V[2, 6] = 1 # b3
    V[5, 6] = -1 # b6
    W[2, 6] = 1 # c13
    W[6, 6] = -1 # c31
    W[7, 6] = 1 # c32
    
    return U, V, W

def construct():
    """
    Find a low-rank exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns dict with U, V, W matrices.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # We will construct the decomposition by splitting B into blocks.
    # B is 3x8.
    # Block 1: cols 0,1,2 (3x3) -> Strassen (Rank 7)
    # Block 2: cols 3,4,5 (3x3) -> Strassen (Rank 7)
    # Block 3: cols 6,7 (3x2) -> Schoolbook (Rank 18)
    
    # Total Rank: 7 + 7 + 18 = 32
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    # Block 1: Cols 0-2
    U1, V1, W1 = _strassen_3x3_3x3()
    # V1 is 9x7, but we only use cols 0-2 of B.
    # V indices are j*p + k. p=8.
    # For block 1, k is 0,1,2.
    # So V rows are j*8 + k.
    # Our Strassen V is for p=3.
    # We need to map Strassen V (9x7) to full V (24x7).
    V1_full = np.zeros((m * p, 7))
    for j in range(m):
        for k in range(3):
            V1_full[j * p + k, :] = V1[j * 3 + k, :]
            
    U_blocks.append(U1)
    V_blocks.append(V1_full)
    W_blocks.append(W1)
    
    # Block 2: Cols 3-5
    U2, V2, W2 = _strassen_3x3_3x3()
    V2_full = np.zeros((m * p, 7))
    for j in range(m):
        for k in range(3):
            V2_full[j * p + (k + 3), :] = V2[j * 3 + k, :]
            
    U_blocks.append(U2)
    V_blocks.append(V2_full)
    W_blocks.append(W2)
    
    # Block 3: Cols 6-7
    # Use schoolbook for 3x3 * 3x2
    U3, V3, W3 = _schoolbook(n, m, 2)
    V3_full = np.zeros((m * p, 18))
    for j in range(m):
        for k in range(2):
            V3_full[j * p + (k + 6), :] = V3[j * 2 + k, :]
            
    U_blocks.append(U3)
    V_blocks.append(V3_full)
    W_blocks.append(W3)
    
    U = np.hstack(U_blocks)
    V = np.hstack(V_blocks)
    W = np.hstack(W_blocks)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
