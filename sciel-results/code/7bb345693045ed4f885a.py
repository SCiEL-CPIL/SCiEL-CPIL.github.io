"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-50 Strassen-based decomposition for 2x4 * 4x7 matrix multiplication.
    Output factor W follows the trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    # Strassen rank for M_{2,2,2} is 7
    # We partition the 4x7 product into:
    #   2 blocks of M_{2,2,4} (each rank 14, since M_{2,2,4} = 2 * M_{2,2,2} = 14)
    #   2 blocks of M_{2,2,3} (each rank 11, since M_{2,2,3} = M_{2,2,2} + M_{2,2,1} = 7 + 4 = 11)
    # Total rank = 2*14 + 2*11 = 28 + 22 = 50
    
    rank = 50
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    def add_strassen_222(U, V, W, a_off, b_off, c_off, col_start):
        """
        Add a 2x2 * 2x2 Strassen block starting at column col_start.
        a_off: row offset in A (0-based, so A row indices are a_off, a_off+1)
        b_off: row offset in B (0-based, so B row indices are b_off, b_off+1)
        c_off: row offset in C output (for the 2x2 C block, C row indices are c_off, c_off+1)
        
        Strassen formulas for M_{2,2,2}:
        M1 = (A11 + A22)(B11 + B22)
        M2 = (A21 + A22) B11
        M3 = A11 (B12 - B22)
        M4 = A22 (B21 - B11)
        M5 = (A11 + A12) B22
        M6 = (A21 - A11)(B11 + B12)
        M7 = (A12 - A22)(B21 + B22)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        
        A indices: A11 = A[a_off, a_off], A12 = A[a_off, a_off+1]
                   A21 = A[a_off+1, a_off], A22 = A[a_off+1, a_off+1]
        B indices: B11 = B[b_off, b_off], B12 = B[b_off, b_off+1]
                   B21 = B[b_off+1, b_off], B22 = B[b_off+1, b_off+1]
        C indices: C11 = C[c_off, c_off], C12 = C[c_off, c_off+1]
                   C21 = C[c_off+1, c_off], C22 = C[c_off+1, c_off+1]
        """
        a11, a12 = a_off, a_off + 1
        a21, a22 = a_off + 1, a_off + 1  # Wait, A22 should be a_off+1, a_off+1
        # Let me redefine:
        a11_row = a_off
        a12_col = a_off + 1  # A12 is row a_off, col a_off+1
        a21_row = a_off + 1
        a21_col = a_off
        a22_row = a_off + 1
        a22_col = a_off + 1
        
        b11_row = b_off
        b11_col = b_off
        b12_row = b_off
        b12_col = b_off + 1
        b21_row = b_off + 1
        b21_col = b_off
        b22_row = b_off + 1
        b22_col = b_off + 1
        
        c11_row = c_off
        c11_col = c_off
        c12_row = c_off
        c12_col = c_off + 1
        c21_row = c_off + 1
        c21_col = c_off
        c22_row = c_off + 1
        c22_col = c_off + 1
        
        # M1 = (A11 + A22)(B11 + B22)
        col = col_start
        U[a11_row * m + 0 + a_off * 0, col] = 0  # This is getting confusing with global indexing
        
        # Let me use a different approach. Let me think about global indexing.
        # U has shape (n*m, rank). Row i corresponds to A element at position i.
        # A is n x m, so A[i,j] has row index i*m + j in U? No wait.
        # Actually, U[i*m+j, r] means row i*m+j of U corresponds to A[i,j].
        # Wait, the problem says U is (n*m, rank). So U row index = i*m + j for A[i,j].
        # V is (m*p, rank). V row index = j*p + k for B[j,k].
        # W is (p*n, rank). W row index = k*n + i for C[i,k] (trace convention W[k*n+i, r]).
        
        # So for A[i,j]: U row = i*m + j
        # For B[j,k]: V row = j*p + k
        # For C[i,k]: W row = k*n + i
        
        # Let me restart this with correct indexing.
        pass
    
    # Let me use a cleaner approach. Build the decomposition directly.
    
    # The matrix multiplication C = A * B where A is 2x4, B is 4x7, C is 2x7.
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    # Partition columns of A (and rows of B) into two groups of 2:
    # Group 0: j = 0, 1
    # Group 1: j = 2, 3
    # Partition columns of B (and thus C) into groups:
    # For the Strassen approach, we need to partition the 7 columns.
    # Let's partition as: cols 0,1 | 2,3 | 4,5 | 6
    # So 4 groups: sizes 2, 2, 2, 1
    
    # For each pair (i_group, k_group) where i_group in {0,1} (rows of A, size 1 each since n=2)
    # and k_group in {0,1,2,3} (column groups of B/C)
    
    # Actually, n=2, so we have 2 rows. m=4, p=7.
    # We can use Strassen for the m dimension and handle p separately.
    
    # Let me think differently. The tensor is T_{i,j,k} = A_{i,j} * B_{j,k} = C_{i,k}
    # with i in [0,1], j in [0,3], k in [0,6].
    
    # Rank 50 approach: Use Strassen on the j-dimension (m=4) with two 2x2 blocks,
    # and for the k-dimension (p=7), use schoolbook for the last column and Strassen-like
    # for pairs.
    
    # Actually, let me just implement it directly using known formulas.
    
    # For a 2x4 * 4x7 matrix product, we can decompose it as:
    # C[:, 0:4] = A[:, 0:2] * B[0:2, 0:4] + A[:, 2:4] * B[2:4, 0:4]
    # C[:, 4:7] = A[:, 0:2] * B[0:2, 4:7] + A[:, 2:4] * B[2:4, 4:7]
    
    # For each 2x2 * 2x2 or 2x2 * 2x3 or 2x2 * 2x1 subproblem, use Strassen where applicable.
    
    # Let me define helper functions to add blocks.
    
    def add_block_22x22(U, V, W, i_off, j_off, k_off, col_start):
        """
        Add Strassen M_{2,2,2} for A[i_off:i_off+2, j_off:j_off+2] * B[j_off:j_off+2, k_off:k_off+2]
        contributing to C[i_off:i_off+2, k_off:k_off+2]
        """
        # A indices: (i_off, j_off), (i_off, j_off+1), (i_off+1, j_off), (i_off+1, j_off+1)
        # B indices: (j_off, k_off), (j_off, k_off+1), (j_off+1, k_off), (j_off+1, k_off+1)
        # C indices: (i_off, k_off), (i_off, k_off+1), (i_off+1, k_off), (i_off+1, k_off+1)
        
        # U row for A[i,j] = i*m + j = i*4 + j
        # V row for B[j,k] = j*p + k = j*7 + k
        # W row for C[i,k] = k*n + i = k*2 + i
        
        a00 = i_off * 4 + j_off
        a01 = i_off * 4 + j_off + 1
        a10 = (i_off + 1) * 4 + j_off
        a11 = (i_off + 1) * 4 + j_off + 1
        
        b00 = j_off * 7 + k_off
        b01 = j_off * 7 + k_off + 1
        b10 = (j_off + 1) * 7 + k_off
        b11 = (j_off + 1) * 7 + k_off + 1
        
        c00 = k_off * 2 + i_off
        c01 = k_off * 2 + i_off + 1  # Wait, C[i_off, k_off+1] = (k_off+1)*2 + i_off
        # Correction: W row for C[i,k] = k*n + i = k*2 + i
        c00 = k_off * 2 + i_off  # C[i_off, k_off]
        c01 = (k_off + 1) * 2 + i_off  # C[i_off, k_off+1]
        c10 = k_off * 2 + i_off + 1  # C[i_off+1, k_off]
        c11 = (k_off + 1) * 2 + i_off + 1  # C[i_off+1, k_off+1]
        
        # M1 = (A00 + A11)(B00 + B11)
        col = col_start
        U[a00, col] = 1
        U[a11, col] = 1
        V[b00, col] = 1
        V[b11, col] = 1
        W[c00, col] = 1
        W[c11, col] = 1
        col += 1
        
        # M2 = (A10 + A11) B00
        U[a10, col] = 1
        U[a11, col] = 1
        V[b00, col] = 1
        W[c10, col] = 1
        col += 1
        
        # M3 = A00 (B01 - B11)
        U[a00, col] = 1
        V[b01, col] = 1
        V[b11, col] = -1
        W[c01, col] = 1
        col += 1
        
        # M4 = A11 (B10 - B00)
        U[a11, col] = 1
        V[b10, col] = 1
        V[b00, col] = -1
        W[c00, col] = 1
        W[c10, col] = 1
        col += 1
        
        # M5 = (A00 + A01) B11
        U[a00, col] = 1
        U[a01, col] = 1
        V[b11, col] = 1
        W[c00, col] = -1
        W[c01, col] = 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[a10, col] = 1
        U[a00, col] = -1
        V[b00, col] = 1
        V[b01, col] = 1
        W[c11, col] = 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[a01, col] = 1
        U[a11, col] = -1
        V[b10, col] = 1
        V[b11, col] = 1
        W[c00, col] = 1
        W[c11, col] = -1
        col += 1
        
        return col
    
    def add_block_22x21(U, V, W, i_off, j_off, k_off, col_start):
        """
        Add schoolbook M_{2,2,1} for A[i_off:i_off+2, j_off:j_off+2] * B[j_off:j_off+2, k_off:k_off+1]
        contributing to C[i_off:i_off+2, k_off:k_off+1]
        Rank = 4
        """
        col = col_start
        for i in range(2):
            for j in range(2):
                U[(i_off + i) * 4 + (j_off + j), col] = 1
                V[(j_off + j) * 7 + k_off, col] = 1
                W[k_off * 2 + (i_off + i), col] = 1
                col += 1
        return col
    
    # Now compose the full decomposition.
    # A is 2x4, B is 4x7, C is 2x7.
    # Split j (columns of A / rows of B) into two groups of 2: j in {0,1} and j in {2,3}
    # Split k (columns of B / C) into groups: {0,1}, {2,3}, {4,5}, {6}
    
    col = 0
    for i_off in range(2):  # 2 rows of A
        for j_group, j_off in enumerate([0, 2]):  # j groups: 0 and 2
            for k_group, k_start, k_len in enumerate([(0, 0, 2), (2, 2, 2), (4, 4, 2), (6, 6, 1)]):
                if k_len == 2:
                    col = add_block_22x22(U, V, W, i_off, j_off, k_start, col)
                else:  # k_len == 1
                    col = add_block_22x21(U, V, W, i_off, j_off, k_start, col)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
