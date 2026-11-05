"""
Program for finding a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 5 matrix.
The output factor follows the official trace convention W[k*n+i, r].
"""
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication.
    
    Returns:
        dict: A dictionary with keys 'U', 'V', 'W' containing the factor matrices.
              U has shape (n*m, rank) = (6, rank)
              V has shape (m*p, rank) = (15, rank)
              W has shape (p*n, rank) = (10, rank)
    """
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2
    # The 7 rank-1 tensors for 2x2 matrix multiplication
    # C = A * B
    # A = [[a0, a1], [a2, a3]]
    # B = [[b0, b1], [b2, b3]]
    # C = [[c0, c1], [c2, c3]]
    #
    # Strassen's 7 products:
    # M1 = (a0 + a3)(b0 + b3)
    # M2 = (a2 + a3)b0
    # M3 = a0(b1 - b3)
    # M4 = a3(b2 - b0)
    # M5 = (a0 + a1)b3
    # M6 = (a2 + a3)(b1 + b2)
    # M7 = (a1 + a3)(b0 + b2)
    #
    # c0 = M1 - M4 + M7 - M5
    # c1 = M3 + M5
    # c2 = M2 + M4
    # c3 = M1 - M2 + M6 - M7
    #
    # U factors (for A):
    # M1: [1, 0, 0, 1]
    # M2: [0, 0, 1, 1]
    # M3: [1, 0, 0, 0]
    # M4: [0, 0, 0, 1]
    # M5: [1, 1, 0, 0]
    # M6: [0, 0, 1, 1]
    # M7: [0, 1, 0, 1]
    #
    # V factors (for B):
    # M1: [1, 0, 0, 1]
    # M2: [1, 0, 0, 0]
    # M3: [0, 1, 0, -1]
    # M4: [-1, 0, 1, 0]
    # M5: [0, 0, 0, 1]
    # M6: [0, 1, 1, 0]
    # M7: [1, 0, 1, 0]
    #
    # W factors (for C):
    # M1: [1, 0, 0, -1]
    # M2: [0, 0, -1, 1]
    # M3: [0, 1, 0, 0]
    # M4: [-1, 0, 1, 0]
    # M5: [-1, 1, 0, 0]
    # M6: [0, 0, 0, -1]
    # M7: [-1, 0, 0, 1]
    
    strassen_U = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [0, 0, 1, 1],  # M6
        [0, 1, 0, 1],  # M7
    ]
    
    strassen_V = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1],  # M5
        [0, 1, 1, 0],  # M6
        [1, 0, 1, 0],  # M7
    ]
    
    strassen_W = [
        [1, 0, 0, -1],  # M1
        [0, 0, -1, 1],  # M2
        [0, 1, 0, 0],   # M3
        [-1, 0, 1, 0],  # M4
        [-1, 1, 0, 0],  # M5
        [0, 0, 0, -1],  # M6
        [-1, 0, 0, 1],  # M7
    ]
    
    # Block decomposition:
    # Block 1: A rows 0-1, cols 0-1 * B rows 0-1, cols 0-1 -> C rows 0-1, cols 0-1
    #   Uses A indices: 0,1,2,3 (i*m+j for i in {0,1}, j in {0,1})
    #   Uses B indices: 0,1,2,3 (j*p+k for j in {0,1}, k in {0,1})
    #   Produces C indices: 0,1,2,3 (k*n+i for k in {0,1}, i in {0,1})
    #   Rank: 7 (Strassen)
    #
    # Block 2: A rows 0-1, cols 0-1 * B rows 0-1, cols 2-3 -> C rows 0-1, cols 2-3
    #   Uses A indices: 0,1,2,3
    #   Uses B indices: 2,3,4,5 (j*p+k for j in {0,1}, k in {2,3})
    #   Produces C indices: 4,5,6,7 (k*n+i for k in {2,3}, i in {0,1})
    #   Rank: 7 (Strassen)
    #
    # Block 3: A rows 0-1, cols 0-1 * B rows 0-1, cols 4 -> C rows 0-1, cols 4
    #   Uses A indices: 0,1,2,3
    #   Uses B indices: 6,7 (j*p+k for j in {0,1}, k in {4})
    #   Produces C indices: 8,9 (k*n+i for k in {4}, i in {0,1})
    #   Rank: 4 (schoolbook for 2x2x1)
    #
    # Block 4: A rows 0-1, cols 2 * B rows 2, cols 0-4 -> C rows 0-1, cols 0-4
    #   Uses A indices: 4,5 (i*m+j for i in {0,1}, j in {2})
    #   Uses B indices: 10,11,12,13,14 (j*p+k for j in {2}, k in {0,1,2,3,4})
    #   Produces C indices: 0,1,2,3,4,5,6,7,8,9 (k*n+i for k in {0,1,2,3,4}, i in {0,1})
    #   Rank: 10 (schoolbook for 2x1x5)
    
    total_rank = 7 + 7 + 4 + 10  # = 28
    
    U = [[0] * total_rank for _ in range(n * m)]  # 6 x 28
    V = [[0] * total_rank for _ in range(m * p)]  # 15 x 28
    W = [[0] * total_rank for _ in range(p * n)]  # 10 x 28
    
    col_offset = 0
    
    # Block 1: Strassen 2x2x2
    # A indices: 0,1,2,3
    # B indices: 0,1,2,3
    # C indices: 0,1,2,3
    for r in range(7):
        for a_idx, val in enumerate(strassen_U[r]):
            if val != 0:
                U[a_idx][col_offset + r] = val
        for b_idx, val in enumerate(strassen_V[r]):
            if val != 0:
                V[b_idx][col_offset + r] = val
        for c_idx, val in enumerate(strassen_W[r]):
            if val != 0:
                W[c_idx][col_offset + r] = val
    col_offset += 7
    
    # Block 2: Strassen 2x2x2
    # A indices: 0,1,2,3
    # B indices: 2,3,4,5 (offset by 2)
    # C indices: 4,5,6,7 (offset by 4)
    for r in range(7):
        for a_idx, val in enumerate(strassen_U[r]):
            if val != 0:
                U[a_idx][col_offset + r] = val
        for b_idx, val in enumerate(strassen_V[r]):
            if val != 0:
                V[b_idx + 2][col_offset + r] = val
        for c_idx, val in enumerate(strassen_W[r]):
            if val != 0:
                W[c_idx + 4][col_offset + r] = val
    col_offset += 7
    
    # Block 3: Schoolbook 2x2x1
    # A indices: 0,1,2,3
    # B indices: 6,7
    # C indices: 8,9
    # C[0,4] = A[0,0]*B[0,4] + A[0,1]*B[1,4]
    # C[1,4] = A[1,0]*B[0,4] + A[1,1]*B[1,4]
    # 
    # Term 0: A[0,0] * B[0,4] -> C[0,4]
    U[0][col_offset + 0] = 1
    V[6][col_offset + 0] = 1
    W[8][col_offset + 0] = 1
    # Term 1: A[0,1] * B[1,4] -> C[0,4]
    U[1][col_offset + 1] = 1
    V[7][col_offset + 1] = 1
    W[8][col_offset + 1] = 1
    # Term 2: A[1,0] * B[0,4] -> C[1,4]
    U[2][col_offset + 2] = 1
    V[6][col_offset + 2] = 1
    W[9][col_offset + 2] = 1
    # Term 3: A[1,1] * B[1,4] -> C[1,4]
    U[3][col_offset + 3] = 1
    V[7][col_offset + 3] = 1
    W[9][col_offset + 3] = 1
    col_offset += 4
    
    # Block 4: Schoolbook 2x1x5
    # A indices: 4,5
    # B indices: 10,11,12,13,14
    # C indices: 0,1,2,3,4,5,6,7,8,9
    # C[i,k] = A[i,2] * B[2,k]
    for i in range(2):
        for k in range(5):
            c_idx = k * n + i
            U[i * m + 2][col_offset + i * 5 + k] = 1
            V[2 * p + k][col_offset + i * 5 + k] = 1
            W[c_idx][col_offset + i * 5 + k] = 1
    col_offset += 10
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
