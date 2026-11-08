import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-17 exact tensor decomposition for 2x3 x 3x3 matrix multiplication.
    
    U: shape (6, 17)  - factor for A (2x3)
    V: shape (9, 17)  - factor for B (3x3)
    W: shape (6, 17)  - factor for C (3x2), using trace convention C[k*n+i, r]
    
    The decomposition combines:
    - Strassen's 7-term algorithm for the 2x2 block multiply
    - 10 additional terms for the remaining columns/rows
    """
    
    n, m, p = 2, 3, 3
    rank = 17
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    # Indexing conventions:
    # A[i,j]: i in [0,1], j in [0,2] -> index i*3+j in U
    # B[j,k]: j in [0,2], k in [0,2] -> index j*3+k in V
    # C[k,i]: k in [0,2], i in [0,1] -> index k*2+i in W (trace convention)
    
    # === STRASSEN PART: 2x2 block multiplication (indices 0-6) ===
    # Multiplies A[:, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Strassen products:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00       -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11)       -> C01 += M3, C11 += M3
    # M4 = A11(B10-B00)       -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11       -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    
    # M1
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # === COLUMNS FROM B[2,:]: A[:,2] * B[2,0:2] -> C[0:2, 0:2] (indices 7-10) ===
    
    # r=7: A[0,2] * B[2,0] -> C[0,0]
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    # r=8: A[0,2] * B[2,1] -> C[0,1]
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1
    
    # r=9: A[1,2] * B[2,0] -> C[1,0]
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1
    
    # r=10: A[1,2] * B[2,1] -> C[1,1]
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1
    
    # === COLUMN 2 OF C: A[:,:2] * B[:,2] -> C[:,2] (indices 11-16) ===
    
    # r=11: A[0,0] * B[0,2] -> C[0,2]
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1
    
    # r=12: A[0,1] * B[1,2] -> C[0,2]
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1
    
    # r=13: A[0,2] * B[2,2] -> C[0,2]
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1
    
    # r=14: A[1,0] * B[0,2] -> C[1,2]
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1
    
    # r=15: A[1,1] * B[1,2] -> C[1,2]
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1
    
    # r=16: A[1,2] * B[2,2] -> C[1,2]
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
