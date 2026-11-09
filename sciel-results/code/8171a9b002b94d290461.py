"""Matrix multiplication 3x3x4 with Strassen-style decomposition (rank 34)."""
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 4
    rank = 34

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Block decomposition:
    # A is 3x3, split rows into {0,1} and {2}
    # B is 3x4, split cols into {0,1} and {2,3}
    # C is 3x4, split rows into {0,1} and {2}, cols into {0,1} and {2,3}
    #
    # C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2] + A[0:2, 2] * B[2, 0:2]
    # C[0:2, 2:4] = A[0:2, 0:2] * B[0:2, 2:4] + A[0:2, 2] * B[2, 2:4]
    # C[2, 0:2]   = A[2, 0:2] * B[0:2, 0:2] + A[2, 2] * B[2, 0:2]
    # C[2, 2:4]   = A[2, 0:2] * B[0:2, 2:4] + A[2, 2] * B[2, 2:4]
    #
    # Each 2x2x2 block uses Strassen (7 mults)
    # Each 2x1x2 block uses schoolbook (4 mults)
    # Each 1x2x2 block uses schoolbook (4 mults)
    # Each 1x1x2 block uses schoolbook (2 mults)
    #
    # Total: 2*7 + 2*4 + 2*4 + 2*2 = 14 + 8 + 8 + 4 = 34

    col = 0

    # Strassen for 2x2x2
    def strassen_222(U, V, W, row_offset_A, col_offset_A, row_offset_B, col_offset_B, row_offset_C, col_offset_C, col_start):
        # A block: rows row_offset_A..row_offset_A+1, cols col_offset_A..col_offset_A+1
        # B block: rows row_offset_B..row_offset_B+1, cols col_offset_B..col_offset_B+1
        # C block: rows row_offset_C..row_offset_C+1, cols col_offset_C..col_offset_C+1
        # U index: i*m + j where i is row in A, j is col in A
        # V index: j*p + k where j is row in B, k is col in B
        # W index: k*n + i where k is col in C, i is row in C
        
        # Strassen:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        def add_U(i, j, val):
            U[i, col_start] += val
        def add_V(j, k, val):
            V[j, col_start] += val
        def add_W(k, i, val):
            W[k, col_start] += val
        
        # M1 = (A11 + A22)(B11 + B22)
        # A11: (row_offset_A, col_offset_A), A22: (row_offset_A+1, col_offset_A+1)
        # B11: (row_offset_B, col_offset_B), B22: (row_offset_B+1, col_offset_B+1)
        U[row_offset_A * m + col_offset_A, col_start] = 1
        U[row_offset_A * m + col_offset_A, col_start] += 1  # A11 appears in M1
        # Wait, this is wrong. Let me redo.
        # Actually, the tensor decomposition is linear. Each "multiplication" M_r corresponds to
        # U_r, V_r, W_r such that C = sum_r W_r * (U_r^T A V_r)
        # For Strassen, each M_r is a product of a linear combination of A entries and a linear combination of B entries.
        # So U_r has entries that are coefficients of A entries, V_r has coefficients of B entries, W_r has coefficients of C entries.
        
        # Let me redo this more carefully.
        # M1 = (A11 + A22) * (B11 + B22)
        # U contribution: A11 and A22 each get coefficient 1
        # V contribution: B11 and B22 each get coefficient 1
        # W contribution: C11 gets +1, C22 gets +1
        
        # But wait, in the tensor decomposition, each term M_r contributes to multiple C entries.
        # C11 = M1 + M4 - M5 + M7
        # So M1 contributes +1 to C11 and +1 to C22.
        
        # Let me define the 7 Strassen terms and their contributions.
        
        # Term 0: M1 = (A11+A22)(B11+B22)
        # U: A11=1, A22=1
        # V: B11=1, B22=1
        # W: C11=+1, C22=+1
        U[row_offset_A * m + col_offset_A, col_start] += 1
        U[(row_offset_A + 1) * m + (col_offset_A + 1), col_start] += 1
        V[row_offset_B * p + col_offset_B, col_start] += 1
        V[(row_offset_B + 1) * p + (col_offset_B + 1), col_start] += 1
        W[col_offset_C * n + row_offset_C, col_start] += 1
        W[(col_offset_C + 1) * n + (row_offset_C + 1), col_start] += 1
        col_start += 1
        
        # Term 1: M2 = (A21+A22)B11
        # U: A21=1, A22=1
        # V: B11=1
        # W: C21=+1, C22=-1
        U[(row_offset_A + 1) * m + col_offset_A, col_start] += 1
        U[(row_offset_A + 1) * m + (col_offset_A + 1), col_start] += 1
        V[row_offset_B * p + col_offset_B, col_start] += 1
        W[col_offset_C * n + (row_offset_C + 1), col_start] += 1
        W[(col_offset_C + 1) * n + (row_offset_C + 1), col_start] -= 1
        col_start += 1
        
        # Term 2: M3 = A11(B21-B22)
        # U: A11=1
        # V: B21=1, B22=-1
        # W: C12=+1, C22=+1
        U[row_offset_A * m + col_offset_A, col_start] += 1
        V[(row_offset_B + 1) * p + col_offset_B, col_start] += 1
        V[(row_offset_B + 1) * p + (col_offset_B + 1), col_start] -= 1
        W[(col_offset_C + 1) * n + row_offset_C, col_start] += 1
        W[(col_offset_C + 1) * n + (row_offset_C + 1), col_start] += 1
        col_start += 1
        
        # Term 3: M4 = A22(B12-B11)
        # U: A22=1
        # V: B12=1, B11=-1
        # W: C11=+1, C21=+1
        U[(row_offset_A + 1) * m + (col_offset_A + 1), col_start] += 1
        V[row_offset_B * p + (col_offset_B + 1), col_start] += 1
        V[row_offset_B * p + col_offset_B, col_start] -= 1
        W[col_offset_C * n + row_offset_C, col_start] += 1
        W[col_offset_C * n + (row_offset_C + 1), col_start] += 1
        col_start += 1
        
        # Term 4: M5 = (A11+A12)B22
        # U: A11=1, A12=1
        # V: B22=1
        # W: C11=-1, C12=+1
        U[row_offset_A * m + col_offset_A, col_start] += 1
        U[row_offset_A * m + (col_offset_A + 1), col_start] += 1
        V[(row_offset_B + 1) * p + (col_offset_B + 1), col_start] += 1
        W[col_offset_C * n + row_offset_C, col_start] -= 1
        W[(col_offset_C + 1) * n + row_offset_C, col_start] += 1
        col_start += 1
        
        # Term 5: M6 = (A21-A11)(B11+B12)
        # U: A21=1, A11=-1
        # V: B11=1, B12=1
        # W: C22=+1
        U[(row_offset_A + 1) * m + col_offset_A, col_start] += 1
        U[row_offset_A * m + col_offset_A, col_start] -= 1
        V[row_offset_B * p + col_offset_B, col_start] += 1
        V[row_offset_B * p + (col_offset_B + 1), col_start] += 1
        W[(col_offset_C + 1) * n + (row_offset_C + 1), col_start] += 1
        col_start += 1
        
        # Term 6: M7 = (A12-A22)(B21+B22)
        # U: A12=1, A22=-1
        # V: B21=1, B22=1
        # W: C11=+1
        U[row_offset_A * m + (col_offset_A + 1), col_start] += 1
        U[(row_offset_A + 1) * m + (col_offset_A + 1), col_start] -= 1
        V[(row_offset_B + 1) * p + col_offset_B, col_start] += 1
        V[(row_offset_B + 1) * p + (col_offset_B + 1), col_start] += 1
        W[col_offset_C * n + row_offset_C, col_start] += 1
        col_start += 1
        
        return col_start
    
    # Schoolbook for 2x1x2 (A is 2x1, B is 1x2, C is 2x2)
    def schoolbook_212(U, V, W, row_offset_A, col_offset_A, row_offset_B, col_offset_B, row_offset_C, col_offset_C, col_start):
        # C[i,k] = A[i, col_offset_A] * B[row_offset_B, k]
        # 4 terms
        for i in range(2):
            for k in range(2):
                U[(row_offset_A + i) * m + col_offset_A, col_start] += 1
                V[row_offset_B * p + (col_offset_B + k), col_start] += 1
                W[(col_offset_C + k) * n + (row_offset_C + i), col_start] += 1
                col_start += 1
        return col_start
    
    # Schoolbook for 1x2x2 (A is 1x2, B is 2x2, C is 1x2)
    def schoolbook_122(U, V, W, row_offset_A, col_offset_A, row_offset_B, col_offset_B, row_offset_C, col_offset_C, col_start):
        # C[k] = sum_j A[col_offset_A+j] * B[j, k]
        # 4 terms
        for j in range(2):
            for k in range(2):
                U[row_offset_A * m + (col_offset_A + j), col_start] += 1
                V[(row_offset_B + j) * p + (col_offset_B + k), col_start] += 1
                W[(col_offset_C + k) * n + row_offset_C, col_start] += 1
                col_start += 1
        return col_start
    
    # Schoolbook for 1x1x2 (A is 1x1, B is 1x2, C is 1x2)
    def schoolbook_112(U, V, W, row_offset_A, col_offset_A, row_offset_B, col_offset_B, row_offset_C, col_offset_C, col_start):
        # C[k] = A[row_offset_A, col_offset_A] * B[row_offset_B, k]
        # 2 terms
        for k in range(2):
            U[row_offset_A * m + col_offset_A, col_start] += 1
            V[row_offset_B * p + (col_offset_B + k), col_start] += 1
            W[(col_offset_C + k) * n + row_offset_C, col_start] += 1
            col_start += 1
        return col_start

    # Now apply to our blocks:
    # A rows: 0,1 (top) and 2 (bottom)
    # A cols: 0,1 (left) and 2 (right)
    # B rows: 0,1 (top) and 2 (bottom)
    # B cols: 0,1 (left) and 2,3 (right)
    # C rows: 0,1 (top) and 2 (bottom)
    # C cols: 0,1 (left) and 2,3 (right)

    # Block 1: C[0:2, 0:2] from A[0:2, 0:2] * B[0:2, 0:2] (2x2x2 Strassen)
    col = strassen_222(U, V, W, 0, 0, 0, 0, 0, 0, col)
    
    # Block 2: C[0:2, 0:2] from A[0:2, 2] * B[2, 0:2] (2x1x2 schoolbook)
    col = schoolbook_212(U, V, W, 0, 2, 2, 0, 0, 0, col)
    
    # Block 3: C[0:2, 2:4] from A[0:2, 0:2] * B[0:2, 2:4] (2x2x2 Strassen)
    col = strassen_222(U, V, W, 0, 0, 0, 2, 0, 2, col)
    
    # Block 4: C[0:2, 2:4] from A[0:2, 2] * B[2, 2:4] (2x1x2 schoolbook)
    col = schoolbook_212(U, V, W, 0, 2, 2, 2, 0, 2, col)
    
    # Block 5: C[2, 0:2] from A[2, 0:2] * B[0:2, 0:2] (1x2x2 schoolbook)
    col = schoolbook_122(U, V, W, 2, 0, 0, 0, 2, 0, col)
    
    # Block 6: C[2, 0:2] from A[2, 2] * B[2, 0:2] (1x1x2 schoolbook)
    col = schoolbook_112(U, V, W, 2, 2, 2, 0, 2, 0, col)
    
    # Block 7: C[2, 2:4] from A[2, 0:2] * B[0:2, 2:4] (1x2x2 schoolbook)
    col = schoolbook_122(U, V, W, 2, 0, 0, 2, 2, 2, col)
    
    # Block 8: C[2, 2:4] from A[2, 2] * B[2, 2:4] (1x1x2 schoolbook)
    col = schoolbook_112(U, V, W, 2, 2, 2, 2, 2, 2, col)
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
