import numpy as np
import time

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 4x4 matrix by a 4x8 matrix.
    
    Strategy: Use Strassen's algorithm for the 4x4x4 sub-problems.
    - Split C (4x8) into C_L (4x4) and C_R (4x4)
    - C_L = A * B_L, C_R = A * B_R
    - Each 4x4x4 multiplication uses Strassen: 7 products of 2x2x2
    - Each 2x2x2 product uses Strassen: 7 scalar multiplications (rank 7)
    - Total rank = 2 * 7 * 7 = 98
    
    The decomposition follows the trace convention W[k*n+i, r].
    """
    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2 decomposition
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Indices for 2x2 matrix: 0, 1, 2, 3 (row-major)
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = 1
    
    # M3 = A00(B10-B11)
    U2[0, 2] = 1
    V2[2, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # M4 = A11(B01-B00)
    U2[3, 3] = 1
    V2[1, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B10)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[2, 5] = 1
    W2[3, 5] = 1
    
    # M7 = (A01-A11)(B01+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[1, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Construct 4x4x4 decomposition using Kronecker product
    # U4 = U2 ⊗ U2, V4 = V2 ⊗ V2, W4 = W2 ⊗ W2
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Total rank is 98
    rank = 98
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # First half: C_L = A * B_L
    # U is same for both
    U[:, 0:49] = U4
    
    # V for B_L (columns 0-3 of B)
    V[0:16, 0:49] = V4
    
    # W for C_L (columns 0-3 of C)
    # Trace convention: W[k*n+i, r]
    # For C_L, k is 0..3, i is 0..3
    # Index in W is k*4 + i
    W[0:16, 0:49] = W4
    
    # Second half: C_R = A * B_R
    # U is same
    U[:, 49:98] = U4
    
    # V for B_R (columns 4-7 of B)
    # B is 4x8. B_R is B[:, 4:8].
    # Index in V is k*p + j. For B_R, j is 0..3 relative to B_R, so 4..7 in B.
    # V4 has rows corresponding to j=0..3.
    # We need to place V4 into rows 16..31 of V (since k=0..3, j=4..7 -> 0*8+4=4 ... 3*8+7=31)
    # Wait, V is (m*p, rank) = (32, 98).
    # V[k, j] in B corresponds to row k*8 + j.
    # For B_L (j=0..3), rows are 0..3, 8..11, 16..19, 24..27.
    # For B_R (j=4..7), rows are 4..7, 12..15, 20..23, 28..31.
    # V4 is 16x49. It corresponds to a 4x4 matrix B.
    # The rows of V4 are indexed by (k, j) where k=0..3, j=0..3.
    # In V, the row for B_L[k, j] is k*8 + j.
    # In V, the row for B_R[k, j] is k*8 + (j+4).
    
    # Let's permute V4 to match the row ordering in V.
    # V4 row index r corresponds to k = r // 4, j = r % 4.
    # For B_L, target row in V is k*8 + j.
    # For B_R, target row in V is k*8 + (j+4).
    
    # Create permutation indices
    perm_L = np.array([k * 8 + j for k in range(4) for j in range(4)])
    perm_R = np.array([k * 8 + (j + 4) for k in range(4) for j in range(4)])
    
    V[perm_L, 0:49] = V4
    V[perm_R, 49:98] = V4
    
    # W for C_R (columns 4-7 of C)
    # C is 4x8. C_R is C[:, 4:8].
    # Index in W is k*n + i. For C_R, i is 0..3 relative to C_R, so 4..7 in C.
    # W4 has rows corresponding to (k, i) where k=0..3, i=0..3.
    # In W, the row for C_L[k, i] is k*4 + i.
    # In W, the row for C_R[k, i] is k*4 + (i+4).
    
    perm_WL = np.array([k * 4 + i for k in range(4) for i in range(4)])
    perm_WR = np.array([k * 4 + (i + 4) for k in range(4) for i in range(4)])
    
    W[perm_WL, 0:49] = W4
    W[perm_WR, 49:98] = W4
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
