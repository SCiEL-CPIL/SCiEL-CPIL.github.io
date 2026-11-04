import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = 3, 4, 6
    rank = 66
    
    # Initialize matrices
    U = [[0.0] * rank for _ in range(n * m)] # 12 x 66
    V = [[0.0] * rank for _ in range(m * p)] # 24 x 66
    W = [[0.0] * rank for _ in range(p * n)] # 18 x 66
    
    col_offset = 0
    
    def add_strassen(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        """
        Adds a Strassen decomposition for a 2x2x2 block.
        A is 2x2, B is 2x2, C is 2x2.
        Indices in the global matrices:
        A rows: a_row_start to a_row_start+1
        A cols: a_col_start to a_col_start+1
        B rows: b_row_start to b_row_start+1
        B cols: b_col_start to b_col_start+1
        C rows: c_row_start to c_row_start+1 (in C matrix, but W index is k*n+i)
        C cols: c_col_start to c_col_start+1
        """
        nonlocal col_offset
        
        # Strassen 7 multiplications
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We need to map these to U, V, W entries.
        # U[i*m+j, c] corresponds to A[i,j]
        # V[j*p+k, c] corresponds to B[j,k]
        # W[k*n+i, c] corresponds to C[k,i] -> Note: W[k*n+i, r] is C[i,k] in standard C=AB?
        # Task says: W[k*n+i, r]. Output factor follows official trace convention.
        # Standard: C[i,k] = sum_j A[i,j] B[j,k].
        # Tensor T[i,j,k] = A[i,j] B[j,k].
        # Decomposition: T = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
        # So W[k*n+i, r] multiplies with U and V to form C[i,k].
        
        def get_u_idx(i, j): return i * m + j
        def get_v_idx(j, k): return j * p + k
        def get_w_idx(k, i): return k * n + i
        
        # M1
        r = col_offset
        U[get_u_idx(a_row_start, a_col_start)][r] += 1
        U[get_u_idx(a_row_start+1, a_col_start+1)][r] += 1
        V[get_v_idx(b_row_start, b_col_start)][r] += 1
        V[get_v_idx(b_row_start+1, b_col_start+1)][r] += 1
        # C11 += M1, C22 += M1
        W[get_w_idx(c_col_start, c_row_start)][r] += 1
        W[get_w_idx(c_col_start+1, c_row_start+1)][r] += 1
        col_offset += 1
        
        # M2
        r = col_offset
        U[get_u_idx(a_row_start+1, a_col_start)][r] += 1
        U[get_u_idx(a_row_start+1, a_col_start+1)][r] += 1
        V[get_v_idx(b_row_start, b_col_start)][r] += 1
        # C21 += M2
        W[get_w_idx(c_col_start, c_row_start+1)][r] += 1
        col_offset += 1
        
        # M3
        r = col_offset
        U[get_u_idx(a_row_start, a_col_start)][r] += 1
        V[get_v_idx(b_row_start+1, b_col_start)][r] += 1
        V[get_v_idx(b_row_start+1, b_col_start+1)][r] += -1
        # C12 += M3, C22 += M3
        W[get_w_idx(c_col_start+1, c_row_start)][r] += 1
        W[get_w_idx(c_col_start+1, c_row_start+1)][r] += 1
        col_offset += 1
        
        # M4
        r = col_offset
        U[get_u_idx(a_row_start+1, a_col_start+1)][r] += 1
        V[get_v_idx(b_row_start+1, b_col_start)][r] += 1
        V[get_v_idx(b_row_start, b_col_start)][r] += -1
        # C11 += M4, C21 += M4
        W[get_w_idx(c_col_start, c_row_start)][r] += 1
        W[get_w_idx(c_col_start, c_row_start+1)][r] += 1
        col_offset += 1
        
        # M5
        r = col_offset
        U[get_u_idx(a_row_start, a_col_start)][r] += 1
        U[get_u_idx(a_row_start, a_col_start+1)][r] += 1
        V[get_v_idx(b_row_start+1, b_col_start+1)][r] += 1
        # C11 -= M5, C12 += M5
        W[get_w_idx(c_col_start, c_row_start)][r] += -1
        W[get_w_idx(c_col_start+1, c_row_start)][r] += 1
        col_offset += 1
        
        # M6
        r = col_offset
        U[get_u_idx(a_row_start+1, a_col_start)][r] += 1
        U[get_u_idx(a_row_start, a_col_start)][r] += -1
        V[get_v_idx(b_row_start, b_col_start)][r] += 1
        V[get_v_idx(b_row_start, b_col_start+1)][r] += 1
        # C22 += M6
        W[get_w_idx(c_col_start+1, c_row_start+1)][r] += 1
        col_offset += 1
        
        # M7
        r = col_offset
        U[get_u_idx(a_row_start, a_col_start+1)][r] += 1
        U[get_u_idx(a_row_start+1, a_col_start+1)][r] += -1
        V[get_v_idx(b_row_start+1, b_col_start)][r] += 1
        V[get_v_idx(b_row_start+1, b_col_start+1)][r] += 1
        # C11 += M7, C12 += M7? No.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Check M7 usage:
        # C11 += M7
        W[get_w_idx(c_col_start, c_row_start)][r] += 1
        # Is M7 in C12? No.
        # Wait, Strassen formulas:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So M7 only in C11.
        col_offset += 1

    def add_schoolbook_1x4x2(a_row, b_col_start, c_col_start):
        """
        Adds schoolbook decomposition for 1x4x2 block.
        A is 1x4, B is 4x2, C is 1x2.
        Rank 8.
        """
        nonlocal col_offset
        
        def get_u_idx(i, j): return i * m + j
        def get_v_idx(j, k): return j * p + k
        def get_w_idx(k, i): return k * n + i
        
        # C[0, 0] = sum A[0,j] B[j, 0]
        # C[0, 1] = sum A[0,j] B[j, 1]
        
        for j in range(4):
            # For C[0,0] (k=0)
            r = col_offset
            U[get_u_idx(a_row, j)][r] = 1
            V[get_v_idx(j, b_col_start)][r] = 1
            W[get_w_idx(b_col_start, a_row)][r] = 1
            col_offset += 1
            
            # For C[0,1] (k=1)
            r = col_offset
            U[get_u_idx(a_row, j)][r] = 1
            V[get_v_idx(j, b_col_start + 1)][r] = 1
            W[get_w_idx(b_col_start + 1, a_row)][r] = 1
            col_offset += 1

    # Block 1: C rows 0-1, cols 0-1
    # A rows 0-1, cols 0-1; B rows 0-1, cols 0-1
    add_strassen(0, 0, 0, 0, 0, 0)
    # A rows 0-1, cols 2-3; B rows 2-3, cols 0-1
    add_strassen(0, 2, 2, 0, 0, 0)
    
    # Block 2: C rows 0-1, cols 2-3
    # A rows 0-1, cols 0-1; B rows 0-1, cols 2-3
    add_strassen(0, 0, 0, 2, 0, 2)
    # A rows 0-1, cols 2-3; B rows 2-3, cols 2-3
    add_strassen(0, 2, 2, 2, 0, 2)
    
    # Block 3: C rows 0-1, cols 4-5
    # A rows 0-1, cols 0-1; B rows 0-1, cols 4-5
    add_strassen(0, 0, 0, 4, 0, 4)
    # A rows 0-1, cols 2-3; B rows 2-3, cols 4-5
    add_strassen(0, 2, 2, 4, 0, 4)
    
    # Block 4: C row 2, cols 0-1
    add_schoolbook_1x4x2(2, 0, 0)
    
    # Block 5: C row 2, cols 2-3
    add_schoolbook_1x4x2(2, 2, 2)
    
    # Block 6: C row 2, cols 4-5
    add_schoolbook_1x4x2(2, 4, 4)
    
    return {"U": U, "V": V, "W": W}
