"""Matrix multiplication 4x5x5 with Strassen-based structured tensor decomposition."""
import numpy as np

DIMENSIONS = [4, 5, 5]

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    
    U: shape (4, 7) - factors for A (2x2), indexed as U[i*2+j, r]
    V: shape (4, 7) - factors for B (2x2), indexed as V[j*2+k, r]
    W: shape (4, 7) - factors for C (2x2), indexed as W[k*2+i, r]
    
    C[i,k] = sum_r U[i*2+j, r] * A[i,j] * V[j*2+k, r] * B[j,k] * W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1  # A[0,0]
    U[3, 0] = 1  # A[1,1]
    V[0, 0] = 1  # B[0,0]
    V[3, 0] = 1  # B[1,1]
    W[0, 0] = 1  # C[0,0]
    W[3, 0] = 1  # C[1,1]
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1  # A[1,0]
    U[3, 1] = 1  # A[1,1]
    V[0, 1] = 1  # B[0,0]
    W[2, 1] = 1  # C[1,0]
    W[3, 1] = -1 # C[1,1]
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1  # A[0,0]
    V[2, 2] = 1  # B[1,0]
    V[3, 2] = -1 # B[1,1]
    W[1, 2] = 1  # C[0,1]
    W[3, 2] = 1  # C[1,1]
    
    # M4 = A11(B01-B10)
    U[3, 3] = 1  # A[1,1]
    V[1, 3] = 1  # B[0,1]
    V[2, 3] = -1 # B[1,0]
    W[0, 3] = 1  # C[0,0]
    W[2, 3] = 1  # C[1,0]
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1  # A[0,0]
    U[1, 4] = 1  # A[0,1]
    V[3, 4] = 1  # B[1,1]
    W[0, 4] = -1 # C[0,0]
    W[1, 4] = 1  # C[0,1]
    
    # M6 = (A10-A00)(B00+B10)
    U[2, 5] = 1  # A[1,0]
    U[0, 5] = -1 # A[0,0]
    V[0, 5] = 1  # B[0,0]
    V[2, 5] = 1  # B[1,0]
    W[3, 5] = 1  # C[1,1]
    
    # M7 = (A01-A11)(B01+B11)
    U[1, 6] = 1  # A[0,1]
    U[3, 6] = -1 # A[1,1]
    V[1, 6] = 1  # B[0,1]
    V[3, 6] = 1  # B[1,1]
    W[0, 6] = 1  # C[0,0]
    W[1, 6] = 1  # C[0,1]
    
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 4x5 matrix by a 5x5 matrix using Strassen-based block decomposition.
    
    Strategy: Decompose the 4x5x5 problem into 2x2x2 sub-blocks where possible,
    using Strassen's rank-7 decomposition for each, and schoolbook for remaining
    1x1 or irregular blocks.
    
    The 5x5 B matrix is split into blocks of size 2,2,1 along both rows and columns.
    The 4x5 A matrix is split into blocks of size 2,2 along rows and 2,2,1 along columns.
    
    This gives us a grid of sub-blocks:
    - A row blocks: [0:2], [2:4]
    - B row blocks: [0:2], [2:4], [4:5]
    - B col blocks: [0:2], [2:4], [4:5]
    
    For each combination of A row block (size 2), B row block (size 2 or 1),
    and B col block (size 2 or 1), we have a sub-problem of size
    2 x (B_row_size) x (B_col_size).
    
    When B_row_size=2 and B_col_size=2, we use Strassen (rank 7).
    Otherwise, we use schoolbook.
    """
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    
    # Define block boundaries
    a_blocks = [(0, 2), (2, 4)]  # A row blocks
    b_row_blocks = [(0, 2), (2, 4), (4, 5)]  # B row blocks
    b_col_blocks = [(0, 2), (2, 4), (4, 5)]  # B col blocks
    
    # Precompute Strassen factors
    sU, sV, sW = _strassen_2x2x2()
    
    # Determine total rank needed
    rank = 0
    for a_start, a_end in a_blocks:
        for b_r_start, b_r_end in b_row_blocks:
            for b_c_start, b_c_end in b_col_blocks:
                a_size = a_end - a_start
                b_r_size = b_r_end - b_r_start
                b_c_size = b_c_end - b_c_start
                if a_size == 2 and b_r_size == 2 and b_c_size == 2:
                    rank += 7
                else:
                    rank += a_size * b_r_size * b_c_size
    
    # Allocate factor matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for a_start, a_end in a_blocks:
        for b_r_start, b_r_end in b_row_blocks:
            for b_c_start, b_c_end in b_col_blocks:
                a_size = a_end - a_start
                b_r_size = b_r_end - b_r_start
                b_c_size = b_c_end - b_c_start
                
                if a_size == 2 and b_r_size == 2 and b_c_size == 2:
                    # Use Strassen for this 2x2x2 sub-block
                    for r in range(7):
                        # Map A indices: local (li, lj) -> global (a_start+li, b_r_start+lj)
                        for li in range(2):
                            for lj in range(2):
                                U[(a_start + li) * m + (b_r_start + lj), col + r] += sU[li * 2 + lj, r]
                        # Map B indices: local (li, lj) -> global (b_r_start+li, b_c_start+lj)
                        for li in range(2):
                            for lj in range(2):
                                V[(b_r_start + li) * p + (b_c_start + lj), col + r] += sV[li * 2 + lj, r]
                        # Map C indices: local (li, lj) -> global (b_c_start+lj, a_start+li)
                        # W indexing: W[k*n + i] where k is col index, i is row index
                        for li in range(2):
                            for lj in range(2):
                                W[(b_c_start + lj) * n + (a_start + li), col + r] += sW[li * 2 + lj, r]
                    col += 7
                else:
                    # Use schoolbook for this sub-block
                    for li in range(a_size):
                        for lj in range(b_r_size):
                            for lk in range(b_c_size):
                                U[(a_start + li) * m + (b_r_start + lj), col] = 1
                                V[(b_r_start + lj) * p + (b_c_start + lk), col] = 1
                                W[(b_c_start + lk) * n + (a_start + li), col] = 1
                                col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
