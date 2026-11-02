"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np

def _strassen_2x2x2():
    """
    Return rank-7 Strassen decomposition for 2x2 matrix multiplication.
    Conventions:
      U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
      Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
      Product: C[k,i] = sum_c U[i*2+j, c] * (A[i,j]*B[j,k] expansion)
    Actually, the tensor is T[i,j,k] = A[i,j] * B[j,k].
    Strassen's 7 products:
      M1 = (A00+A11)(B00+B11)
      M2 = (A10+A11)B00
      M3 = A00(B10-B11)
      M4 = A11(B01-B10)
      M5 = (A00+A01)B11
      M6 = (A10-A00)(B00+B10)
      M7 = (A01-A11)(B01+B11)
    Then:
      C00 = M1 + M4 - M5 + M7
      C01 = M3 + M5
      C10 = M2 + M4
      C11 = M1 - M2 + M3 + M6
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Column 0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11
    
    # Column 1: M2 = (A10+A11)B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1  # C10
    W[3, 1] = -1 # C11
    
    # Column 2: M3 = A00(B10-B11)
    U[0, 2] = 1  # A00
    V[2, 2] = 1  # B10
    V[3, 2] = -1 # B11
    W[1, 2] = 1  # C01
    W[3, 2] = 1  # C11
    
    # Column 3: M4 = A11(B01-B10)
    U[3, 3] = 1  # A11
    V[1, 3] = 1  # B01
    V[2, 3] = -1 # B10
    W[0, 3] = 1  # C00
    W[2, 3] = 1  # C10
    
    # Column 4: M5 = (A00+A01)B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    W[0, 4] = -1 # C00
    W[1, 4] = 1  # C01
    
    # Column 5: M6 = (A10-A00)(B00+B10)
    U[0, 5] = -1 # A00
    U[2, 5] = 1  # A10
    V[0, 5] = 1  # B00
    V[2, 5] = 1  # B10
    W[3, 5] = 1  # C11
    
    # Column 6: M7 = (A01-A11)(B01+B11)
    U[1, 6] = 1  # A01
    U[3, 6] = -1 # A11
    V[1, 6] = 1  # B01
    V[3, 6] = 1  # B11
    W[0, 6] = 1  # C00
    
    return U, V, W

def _block_multiply_2x2x6():
    """
    Decompose multiplication of a 2x2 matrix by a 2x6 matrix.
    We can view this as 3 separate 2x2 multiplications (one for each pair of columns of B),
    but since B is 2x6, we split B into three 2x2 blocks.
    Actually, it's easier to use the 2x2x2 Strassen decomposition for each 2x2 block of B.
    So we have 3 blocks of 2x2x2, total rank 21.
    
    U: 4 rows (2x2 A), 21 cols
    V: 12 rows (2x6 B), 21 cols
    W: 12 rows (2x6 C), 21 cols
    """
    U_222, V_222, W_222 = _strassen_2x2x2()
    rank = 21
    U = np.zeros((4, rank))
    V = np.zeros((12, rank))
    W = np.zeros((12, rank))
    
    for block in range(3):
        start = block * 7
        end = (block + 1) * 7
        U[:, start:end] = U_222
        # V rows: block b uses rows 2*b to 2*b+1 of B (which are rows 2*b, 2*b+1 in the 2x6 B)
        V[2*block:2*block+2, start:end] = V_222
        # W rows: block b uses rows 2*b to 2*b+1 of C
        W[2*block:2*block+2, start:end] = W_222
        
    return U, V, W

def _schoolbook_2x2x2():
    """
    Rank-4 schoolbook decomposition for 2x2x2.
    """
    rank = 4
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                c = i*4 + j*2 + k
                U[i*2+j, c] = 1
                V[j*2+k, c] = 1
                W[k*2+i, c] = 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # We split the 4-column middle dimension into two blocks of 2.
    # A is 2x4, split into A1 (2x2) and A2 (2x2)
    # B is 4x6, split into B1 (2x6) and B2 (2x6)
    # C = A1*B1 + A2*B2
    
    # For each block, we multiply a 2x2 by a 2x6.
    # We use Strassen for the 2x2 part, giving rank 7 per 2x2 block of B.
    # Since B is 2x6, we have 3 such blocks, so rank 21 per block.
    # Two blocks total: rank 42.
    
    U_block, V_block, W_block = _block_multiply_2x2x6()
    
    rank = 42
    U = np.zeros((n*m, rank))  # 8 rows
    V = np.zeros((m*p, rank))  # 24 rows
    W = np.zeros((p*n, rank))  # 12 rows
    
    # Block 1: A1 (rows 0-3 of U) * B1 (rows 0-11 of V) -> C (rows 0-11 of W)
    U[0:4, 0:21] = U_block
    V[0:12, 0:21] = V_block
    W[0:12, 0:21] = W_block
    
    # Block 2: A2 (rows 4-7 of U) * B2 (rows 12-23 of V) -> C (rows 0-11 of W)
    U[4:8, 21:42] = U_block
    V[12:24, 21:42] = V_block
    W[0:12, 21:42] = W_block  # Adds to the same C rows
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
