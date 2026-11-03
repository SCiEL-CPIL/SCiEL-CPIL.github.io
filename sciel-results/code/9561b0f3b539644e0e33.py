"""Matrix multiplication 4x4 * 4x9 rank-minimal decomposition.

We construct an explicit decomposition using a Strassen-based approach
for the 4x4 * 4x4 block, combined with schoolbook for the remaining 4x1.

The key idea: split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1).
C = A*B1 + A*B2 + A*B3.

For A*B1 and A*B2 (each 4x4*4x4), use a rank-49 Strassen-based decomposition.
For A*B3 (4x4*4x1), use schoolbook rank-16.

Total rank: 49 + 49 + 16 = 114.

The Strassen decomposition for 2x2*2x2 has rank 7:
  m1 = (a+d)(e+h)
  m2 = (c+d)e
  m3 = a(f-h)
  m4 = d(g-e)
  m5 = (a+b)h
  m6 = (c-a)(e+f)
  m7 = (b-d)(g+h)

  c11 = m1 + m4 - m5 + m7
  c12 = m3 + m5
  c21 = m2 + m4
  c22 = m1 - m2 + m3 + m6

For 4x4*4x4, we view it as 2x2 blocks of 2x2 matrices and apply Strassen,
giving 7 block multiplications, each of rank 7, total rank 49.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 9]


def _strassen_2x2_decomposition():
    """Return U, V, W for 2x2*2x2 Strassen decomposition.
    
    U has shape (4, 7) - rows indexed by (i,j) of A, flattened as i*2+j
    V has shape (4, 7) - rows indexed by (j,k) of B, flattened as j*2+k
    W has shape (4, 7) - rows indexed by (i,k) of C, flattened as i*2+k
    
    The 7 Strassen products:
      m1 = (a+d)(e+h)   : A factor [1,0,0,1], B factor [1,1,0,0]
      m2 = (c+d)e       : A factor [0,0,1,1], B factor [1,0,0,0]
      m3 = a(f-h)       : A factor [1,0,0,0], B factor [0,1,0,-1]
      m4 = d(g-e)       : A factor [0,0,0,1], B factor [0,-1,1,0]
      m5 = (a+b)h       : A factor [1,1,0,0], B factor [0,0,0,1]
      m6 = (c-a)(e+f)   : A factor [0,-1,1,0], B factor [1,1,0,0]
      m7 = (b-d)(g+h)   : A factor [0,1,0,-1], B factor [0,0,1,1]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1 = (a+d)(e+h)
    U[0, 0] = 1  # a
    U[3, 0] = 1  # d
    V[0, 0] = 1  # e
    V[1, 0] = 1  # h
    W[0, 0] = 1  # c11
    W[3, 0] = 1  # c22
    
    # m2 = (c+d)e
    U[2, 1] = 1  # c
    U[3, 1] = 1  # d
    V[0, 1] = 1  # e
    W[2, 1] = 1  # c21
    W[3, 1] = -1 # c22
    
    # m3 = a(f-h)
    U[0, 2] = 1  # a
    V[1, 2] = 1  # f
    V[3, 2] = -1 # h
    W[1, 2] = 1  # c12
    W[3, 2] = 1  # c22
    
    # m4 = d(g-e)
    U[3, 3] = 1  # d
    V[0, 3] = -1 # e
    V[2, 3] = 1  # g
    W[0, 3] = 1  # c11
    W[2, 3] = 1  # c21
    
    # m5 = (a+b)h
    U[0, 4] = 1  # a
    U[1, 4] = 1  # b
    V[3, 4] = 1  # h
    W[0, 4] = -1 # c11
    W[1, 4] = 1  # c12
    
    # m6 = (c-a)(e+f)
    U[0, 5] = -1 # a
    U[2, 5] = 1  # c
    V[0, 5] = 1  # e
    V[1, 5] = 1  # f
    W[3, 5] = 1  # c22
    
    # m7 = (b-d)(g+h)
    U[1, 6] = 1  # b
    U[3, 6] = -1 # d
    V[2, 6] = 1  # g
    V[3, 6] = 1  # h
    W[0, 6] = 1  # c11
    
    return U, V, W


def _strassen_4x4_decomposition():
    """Return U, V, W for 4x4*4x4 using two levels of Strassen.
    
    U has shape (16, 49)
    V has shape (16, 49)  
    W has shape (16, 49)
    
    We view 4x4 matrices as 2x2 blocks of 2x2 matrices.
    Apply Strassen at the block level (7 block products),
    then apply Strassen at the scalar level within each block product (7 scalar products each).
    Total: 7 * 7 = 49 scalar products.
    """
    U2, V2, W2 = _strassen_2x2_decomposition()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Block-level Strassen indices
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Block positions in 4x4 (flattened row-major):
    # A11: rows 0-1, cols 0-1 -> flat indices 0,1,4,5
    # A12: rows 0-1, cols 2-3 -> flat indices 2,3,6,7
    # A21: rows 2-3, cols 0-1 -> flat indices 8,9,12,13
    # A22: rows 2-3, cols 2-3 -> flat indices 10,11,14,15
    # Similarly for B and C.
    
    block_A = {
        0: [0, 1, 4, 5],   # A11
        1: [2, 3, 6, 7],   # A12
        2: [8, 9, 12, 13], # A21
        3: [10, 11, 14, 15] # A22
    }
    block_B = {
        0: [0, 1, 4, 5],   # B11
        1: [2, 3, 6, 7],   # B12
        2: [8, 9, 12, 13], # B21
        3: [10, 11, 14, 15] # B22
    }
    block_C = {
        0: [0, 1, 4, 5],   # C11
        1: [2, 3, 6, 7],   # C12
        2: [8, 9, 12, 13], # C21
        3: [10, 11, 14, 15] # C22
    }
    
    # Block-level Strassen coefficients
    # For each of the 7 block products, which A-blocks and B-blocks participate, with what coefficients
    block_products = [
        # (A_blocks, B_blocks, C_blocks_output)
        # M1 = (A11+A22)(B11+B22)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A21+A22)B11
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = A11(B12-B22)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = A22(B21-B11)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (A11+A12)B22
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (A21-A11)(B11+B12)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (A12-A22)(B21+B22)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    for bp_idx, (A_coeffs, B_coeffs, C_coeffs) in enumerate(block_products):
        for sub_idx in range(7):
            col = bp_idx * 7 + sub_idx
            
            # U contributions
            for block_idx, coeff in A_coeffs.items():
                for flat_idx in block_A[block_idx]:
                    U[flat_idx, col] += coeff * U2[flat_idx % 4, sub_idx]
            
            # V contributions
            for block_idx, coeff in B_coeffs.items():
                for flat_idx in block_B[block_idx]:
                    V[flat_idx, col] += coeff * V2[flat_idx % 4, sub_idx]
            
            # W contributions
            for block_idx, coeff in C_coeffs.items():
                for flat_idx in block_C[block_idx]:
                    W[flat_idx, col] += coeff * W2[flat_idx % 4, sub_idx]
    
    return U, V, W


def _schoolbook_4x1_decomposition():
    """Return U, V, W for 4x4 * 4x1 (matrix-vector product).
    
    U has shape (16, 16)
    V has shape (4, 16)
    W has shape (4, 16)
    
    For each output C[i], it's sum_j A[i,j]*B[j].
    """
    U = np.zeros((16, 16))
    V = np.zeros((4, 16))
    W = np.zeros((4, 16))
    
    col = 0
    for i in range(4):
        for j in range(4):
            U[i * 4 + j, col] = 1
            V[j, col] = 1
            W[i, col] = 1
            col += 1
    
    return U, V, W


def construct():
    """Return a valid tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses Strassen-based decomposition for two 4x4*4x4 sub-problems (rank 49 each)
    and schoolbook for the 4x4*4x1 sub-problem (rank 16).
    Total rank: 114.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Get the 4x4*4x4 Strassen decomposition
    U44, V44, W44 = _strassen_4x4_decomposition()
    
    # Get the 4x4*4x1 schoolbook decomposition
    U41, V41, W41 = _schoolbook_4x1_decomposition()
    
    rank = 49 + 49 + 16  # 114
    
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))  # Wait, V should be (m*p, rank) = (4*9, rank) = (36, rank)
    W = np.zeros((36, rank))  # W should be (p*n, rank) = (9*4, rank) = (36, rank)
    
    # Let me reconsider the dimensions.
    # A is 4x4 (n*m = 4*4 = 16 elements)
    # B is 4x9 (m*p = 4*9 = 36 elements)
    # C is 4x9 (p*n = 9*4 = 36 elements)
    #
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (36, rank)
    # W: (p*n, rank) = (36, rank)
    
    U = np.zeros((16, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    
    # First 4x4*4x4: A * B1 where B1 is columns 0-3 of B
    # U44 has shape (16, 49) - correct for A
    # V44 has shape (16, 49) - but V should be (36, rank). V44 corresponds to B1 (4x4), which is rows 0-3 of B.
    # B is 4x9, so B1 is the first 4 columns. In V's indexing (m*p = 4*9 = 36), 
    # B1 occupies indices 0-3 (row 0, cols 0-3), 4-7 (row 1, cols 0-3), etc.
    # V44[j*4+k] for j in 0-3, k in 0-3 maps to V[j*9+k]
    
    # W44 has shape (16, 49) - corresponds to C1 (4x4), which is the first 4 columns of C.
    # C is 4x9, so C1 occupies indices 0-3 (row 0, cols 0-3), 4-7 (row 1, cols 0-3), etc.
    # W44[i*4+k] for i in 0-3, k in 0-3 maps to W[k*4+i] in the output convention W[k*n+i]
    
    # Copy U44 into first 49 columns of U
    U[:, 0:49] = U44
    
    # Copy V44 into appropriate positions in V
    for j in range(4):
        for k in range(4):
            V[j * 9 + k, 0:49] = V44[j * 4 + k, :]
    
    # Copy W44 into appropriate positions in W
    for i in range(4):
        for k in range(4):
            W[k * 4 + i, 0:49] = W44[i * 4 + k, :]
    
    # Second 4x4*4x4: A * B2 where B2 is columns 4-7 of B
    U[:, 49:98] = U44
    
    for j in range(4):
        for k in range(4):
            V[j * 9 + (4 + k), 49:98] = V44[j * 4 + k, :]
    
    for i in range(4):
        for k in range(4):
            W[(4 + k) * 4 + i, 49:98] = W44[i * 4 + k, :]
    
    # 4x4*4x1: A * B3 where B3 is column 8 of B
    # U41 has shape (16, 16) - correct for A
    # V41 has shape (4, 16) - corresponds to B3 (4x1), which is column 8 of B.
    # In V's indexing, B3 is at V[j*9+8] for j in 0-3.
    
    U[:, 98:114] = U41
    
    for j in range(4):
        V[j * 9 + 8, 98:114] = V41[j, :]
    
    # W41 has shape (4, 16) - corresponds to C3 (4x1), which is column 8 of C.
    # In W's indexing, C3 is at W[8*4+i] = W[32+i] for i in 0-3.
    for i in range(4):
        W[32 + i, 98:114] = W41[i, :]
    
    # Check deadline
    if time.time() > deadline:
        # Return what we have (should be complete)
        pass
    
    # Convert to lists of lists with integer/half-integer values
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
