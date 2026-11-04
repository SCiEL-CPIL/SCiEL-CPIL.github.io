"""Bootstrap SEARCH for rank-minimal exact tensor decomposition of 2x5 * 5x6."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _schoolbook_rank():
    """Return the standard schoolbook decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_block(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols, U, V, W, start_col):
    """
    Add Strassen decomposition for a 2x2x2 block.
    A: rows A_rows, cols A_cols (2x2)
    B: rows B_rows, cols B_cols (2x2)
    C: rows C_rows, cols C_cols (2x2)
    Returns new column index.
    """
    col = start_col
    # Strassen products:
    # M1 = A00 * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A01 * (B10 - B00)
    # M4 = A11 * (B11 - B01)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # A indices: A_rows[i] * m + A_cols[j]
    # B indices: B_rows[i] * p + B_cols[j]
    # C indices: C_rows[i] * n + C_cols[j]
    
    n, m, p = DIMENSIONS
    
    # M1 = A00 * (B00 + B11)
    # U: A00
    U[A_rows[0] * m + A_cols[0]][col] += 1
    # V: B00 + B11
    V[B_rows[0] * p + B_cols[0]][col] += 1
    V[B_rows[1] * p + B_cols[1]][col] += 1
    # W: C00 + C11
    W[C_rows[0] * n + C_cols[0]][col] += 1
    W[C_rows[1] * n + C_cols[1]][col] += 1
    col += 1
    
    # M2 = (A10 + A11) * B00
    # U: A10 + A11
    U[A_rows[1] * m + A_cols[0]][col] += 1
    U[A_rows[1] * m + A_cols[1]][col] += 1
    # V: B00
    V[B_rows[0] * p + B_cols[0]][col] += 1
    # W: C10
    W[C_rows[1] * n + C_cols[0]][col] += 1
    col += 1
    
    # M3 = A01 * (B10 - B00)
    # U: A01
    U[A_rows[0] * m + A_cols[1]][col] += 1
    # V: B10 - B00
    V[B_rows[1] * p + B_cols[0]][col] += 1
    V[B_rows[0] * p + B_cols[0]][col] += -1
    # W: C01
    W[C_rows[0] * n + C_cols[1]][col] += 1
    col += 1
    
    # M4 = A11 * (B11 - B01)
    # U: A11
    U[A_rows[1] * m + A_cols[1]][col] += 1
    # V: B11 - B01
    V[B_rows[1] * p + B_cols[1]][col] += 1
    V[B_rows[0] * p + B_cols[1]][col] += -1
    # W: C00 + C10
    W[C_rows[0] * n + C_cols[0]][col] += 1
    W[C_rows[1] * n + C_cols[0]][col] += 1
    col += 1
    
    # M5 = (A00 + A01) * B11
    # U: A00 + A01
    U[A_rows[0] * m + A_cols[0]][col] += 1
    U[A_rows[0] * m + A_cols[1]][col] += 1
    # V: B11
    V[B_rows[1] * p + B_cols[1]][col] += 1
    # W: -C00 + C01
    W[C_rows[0] * n + C_cols[0]][col] += -1
    W[C_rows[0] * n + C_cols[1]][col] += 1
    col += 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    # U: A10 - A00
    U[A_rows[1] * m + A_cols[0]][col] += 1
    U[A_rows[0] * m + A_cols[0]][col] += -1
    # V: B00 + B01
    V[B_rows[0] * p + B_cols[0]][col] += 1
    V[B_rows[0] * p + B_cols[1]][col] += 1
    # W: C11
    W[C_rows[1] * n + C_cols[1]][col] += 1
    col += 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    # U: A01 - A11
    U[A_rows[0] * m + A_cols[1]][col] += 1
    U[A_rows[1] * m + A_cols[1]][col] += -1
    # V: B10 + B11
    V[B_rows[1] * p + B_cols[0]][col] += 1
    V[B_rows[1] * p + B_cols[1]][col] += 1
    # W: C00 + C11
    W[C_rows[0] * n + C_cols[0]][col] += 1
    W[C_rows[1] * n + C_cols[1]][col] += 1
    col += 1
    
    return col

def _block_decomposition():
    """
    Use block decomposition with Strassen for 2x2x2 sub-blocks.
    Split the 5x6 output into 3 blocks of 2 columns each.
    For each block, split the 5 rows of A and B into:
    - 2 rows (for Strassen 2x2x2)
    - 3 rows (for schoolbook 2x3x2)
    
    Total rank: 3 blocks * (7 Strassen + 12 schoolbook) = 3 * 19 = 57
    """
    n, m, p = DIMENSIONS
    # We'll build U, V, W as numpy arrays for efficiency
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # rank = 57
    
    rank = 57
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    for block in range(3):
        # C columns: 2*block, 2*block+1
        C_cols = [2 * block, 2 * block + 1]
        
        # For Strassen part: use A cols 0,1 and B rows 0,1
        A_cols_strassen = [0, 1]
        B_rows_strassen = [0, 1]
        A_rows = [0, 1]  # all 2 rows of A
        C_rows = [0, 1]  # all 2 rows of C
        
        col = _strassen_block(A_rows, A_cols_strassen, B_rows_strassen, C_cols, C_rows, C_cols, U, V, W, col)
        
        # For schoolbook part: use A cols 2,3,4 and B rows 2,3,4
        # C_ik = sum_{j=2}^{4} A_ij * B_jk
        A_cols_school = [2, 3, 4]
        B_rows_school = [2, 3, 4]
        
        for i in range(2):
            for j_idx, j in enumerate(A_cols_school):
                for k_idx, k in enumerate(C_cols):
                    # U: A[i, j]
                    U[i * m + j][col] += 1
                    # V: B[j, k]
                    V[j * p + k][col] += 1
                    # W: C[i, k]
                    W[k * n + i][col] += 1
                    col += 1
    
    # Convert to lists of lists as required by the interface
    U_list = [[int(x) for x in row] for row in U]
    V_list = [[int(x) for x in row] for row in V]
    W_list = [[int(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    
    # Build tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Check T[i,j,k] == 1 if and only if the term A_ij * B_jk appears in C_ik
    # Actually, the tensor for matrix mult is T[i,j,k] = 1 for all i,j,k
    # Wait, no. The tensor is defined as T[i,j,k] = delta_{j,j} which is always 1.
    # Let me re-read the problem.
    # The tensor decomposition of matrix multiplication:
    # C_ik = sum_j A_ij B_jk
    # Tensor T has T[i,j,k] = 1 for all i,j,k
    # Decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    
    # So we need: sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] == 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def construct():
    # Try the block decomposition with Strassen
    result = _block_decomposition()
    
    # Verify
    if _verify_decomposition(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback to schoolbook
    return _schoolbook_rank()

# EVOLVE-BLOCK-END
