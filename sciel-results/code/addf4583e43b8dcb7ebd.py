import numpy as np
import time
def construct():
    n, m, p = 4, 5, 5
    deadline = time.time() + 110
    
    # Block decomposition:
    # A (4x5) rows split as [0,1], [2,3]; cols split as [0,1], [2,3,4]
    # B (5x5) rows split as [0,1], [2,3,4]; cols split as [0,1], [2,3,4]
    # C (4x5) rows split as [0,1], [2,3]; cols split as [0,1], [2,3,4]
    
    # Block sizes:
    # A11: rows 0-1, cols 0-1 (2x2)
    # A12: rows 0-1, cols 2-4 (2x3)
    # A21: rows 2-3, cols 0-1 (2x2)
    # A22: rows 2-3, cols 2-4 (2x3)
    
    # B11: rows 0-1, cols 0-1 (2x2)
    # B12: rows 0-1, cols 2-4 (2x3)
    # B21: rows 2-4, cols 0-1 (3x2)
    # B22: rows 2-4, cols 2-4 (3x3)
    
    # C11 = A11*B11 + A12*B21 (2x2)
    # C12 = A11*B12 + A12*B22 (2x3)
    # C21 = A21*B11 + A22*B21 (2x2)
    # C22 = A21*B12 + A22*B22 (2x3)
    
    # Ranks:
    # A11*B11: 2x2x2 -> Strassen rank 7
    # A12*B21: 2x3x2 -> schoolbook rank 12
    # A11*B12: 2x2x3 -> schoolbook rank 12
    # A12*B22: 2x3x3 -> schoolbook rank 18
    # A21*B11: 2x2x2 -> Strassen rank 7
    # A22*B21: 2x3x2 -> schoolbook rank 12
    # A21*B12: 2x2x3 -> schoolbook rank 12
    # A22*B22: 2x3x3 -> schoolbook rank 18
    # Total: 7+12+12+18+7+12+12+18 = 98
    
    total_rank = 98
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    
    # Helper: Strassen decomposition for 2x2x2
    def strassen_2x2x2(U_mat, V_mat, W_mat, row_offset_u, col_offset_v, row_offset_w, start_col):
        """Add Strassen decomposition for a 2x2x2 block."""
        # Strassen's 7 multiplications for 2x2 matrices
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For 2x2x2, the U matrix has 4 entries (2x2), V has 4 entries (2x2), W has 4 entries (2x2)
        # U indices: (0,0), (0,1), (1,0), (1,1) -> flat: 0,1,2,3
        # V indices: (0,0), (0,1), (1,0), (1,1) -> flat: 0,1,2,3
        # W indices: (0,0), (0,1), (1,0), (1,1) -> flat: 0,1,2,3
        
        # Strassen coefficients:
        # Each Mj corresponds to a rank-1 term
        # We need to express U, V, W for each of the 7 terms
        
        # For term j, U_mat has values for the 4 A-entries
        # V_mat has values for the 4 B-entries
        # W_mat has values for the 4 C-entries
        
        # M1 = (A11+A22)(B11+B22)
        # U: A11=1, A12=0, A21=0, A22=1
        # V: B11=1, B12=0, B21=0, B22=1
        # W: C11=1, C12=0, C21=0, C22=1
        U_mat[0, start_col] = 1  # A11
        U_mat[3, start_col] = 1  # A22
        V_mat[0, start_col] = 1  # B11
        V_mat[3, start_col] = 1  # B22
        W_mat[0, start_col] = 1  # C11
        W_mat[3, start_col] = 1  # C22
        col += 1
        
        # M2 = (A21+A22)B11
        U_mat[2, start_col] = 1  # A21
        U_mat[3, start_col] = 1  # A22
        V_mat[0, start_col] = 1  # B11
        W_mat[2, start_col] = 1  # C21
        col += 1
        
        # M3 = A11(B12-B22)
        U_mat[0, start_col] = 1  # A11
        V_mat[1, start_col] = 1  # B12
        V_mat[3, start_col] = -1  # B22
        W_mat[1, start_col] = 1  # C12
        W_mat[3, start_col] = 1  # C22
        col += 1
        
        # M4 = A22(B21-B11)
        U_mat[3, start_col] = 1  # A22
        V_mat[2, start_col] = 1  # B21
        V_mat[0, start_col] = -1  # B11
        W_mat[0, start_col] = 1  # C11
        W_mat[2, start_col] = 1  # C21
        col += 1
        
        # M5 = (A11+A12)B22
        U_mat[0, start_col] = 1  # A11
        U_mat[1, start_col] = 1  # A12
        V_mat[3, start_col] = 1  # B22
        W_mat[0, start_col] = -1  # C11
        W_mat[1, start_col] = 1  # C12
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        U_mat[2, start_col] = 1  # A21
        U_mat[0, start_col] = -1  # A11
        V_mat[0, start_col] = 1  # B11
        V_mat[1, start_col] = 1  # B12
        W_mat[3, start_col] = 1  # C22
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        U_mat[1, start_col] = 1  # A12
        U_mat[3, start_col] = -1  # A22
        V_mat[2, start_col] = 1  # B21
        V_mat[3, start_col] = 1  # B22
        W_mat[0, start_col] = 1  # C11
        W_mat[3, start_col] = -1  # C22
        col += 1
        
        return col
    
    # Helper: Schoolbook decomposition for arbitrary block sizes
    def schoolbook_block(U_mat, V_mat, W_mat, 
                         a_rows, a_cols, b_rows, b_cols, c_rows, c_cols,
                         start_col):
        """Add schoolbook decomposition for a block multiplication."""
        na = len(a_rows)
        ma = len(a_cols)
        nb = len(b_rows)
        pb = len(b_cols)
        nc = len(c_rows)
        pc = len(c_cols)
        
        for i_idx, i in enumerate(a_rows):
            for j_idx, j in enumerate(a_cols):
                for k_idx, k in enumerate(b_cols):
                    # A[i,j] * B[j,k] -> C[i,k]
                    # U index: i*m + j
                    # V index: j*p + k
                    # W index: k*n + i
                    U_mat[i*m + j, start_col] = 1
                    V_mat[j*p + k, start_col] = 1
                    W_mat[k*n + i, start_col] = 1
                    start_col += 1
        
        return start_col
    
    # Now construct the full decomposition
    
    # C11 = A11*B11 + A12*B21
    # A11*B11: 2x2x2, use Strassen
    # A11 rows: [0,1], cols: [0,1]
    # B11 rows: [0,1], cols: [0,1]
    # C11 rows: [0,1], cols: [0,1]
    
    # For Strassen, we need to map the 2x2 blocks to global indices
    # A11: U indices for (row, col) in {0,1}x{0,1}
    #   (0,0) -> U[0*5+0] = U[0]
    #   (0,1) -> U[0*5+1] = U[1]
    #   (1,0) -> U[1*5+0] = U[5]
    #   (1,1) -> U[1*5+1] = U[6]
    
    # Let me create local U, V, W matrices for each block and then map to global
    
    # Actually, it's easier to directly fill the global U, V, W matrices
    
    # A11*B11 (2x2x2) with Strassen:
    # Local A indices: (0,0)->U[0], (0,1)->U[1], (1,0)->U[5], (1,1)->U[6]
    # Local B indices: (0,0)->V[0], (0,1)->V[1], (1,0)->V[5], (1,1)->V[6]
    # Local C indices: (0,0)->W[0*4+0]=W[0], (0,1)->W[1*4+0]=W[4], (1,0)->W[0*4+1]=W[1], (1,1)->W[1*4+1]=W[5]
    # Wait, W index is k*n+i where k is col, i is row
    # C[i,k] -> W[k*n+i]
    # C[0,0] -> W[0*4+0] = W[0]
    # C[0,1] -> W[1*4+0] = W[4]
    # C[1,0] -> W[0*4+1] = W[1]
    # C[1,1] -> W[1*4+1] = W[5]
    
    # Strassen for A11*B11:
    # M1: A11+A22, B11+B22 -> C11, C22
    U[0, col] = 1; U[6, col] = 1  # A[0,0], A[1,1]
    V[0, col] = 1; V[6, col] = 1  # B[0,0], B[1,1]
    W[0, col] = 1; W[5, col] = 1  # C[0,0], C[1,1]
    col += 1
    
    # M2: (A21+A22), B11 -> C21
    U[5, col] = 1; U[6, col] = 1  # A[1,0], A[1,1]
    V[0, col] = 1  # B[0,0]
    W[1, col] = 1  # C[1,0]
    col += 1
    
    # M3: A11, (B12-B22) -> C12, C22
    U[0, col] = 1  # A[0,0]
    V[1, col] = 1; V[6, col] = -1  # B[0,1], B[1,1]
    W[4, col] = 1; W[5, col] = 1  # C[0,1], C[1,1]
    col += 1
    
    # M4: A22, (B21-B11) -> C11, C21
    U[6, col] = 1  # A[1,1]
    V[5, col] = 1; V[0, col] = -1  # B[1,0], B[0,0]
    W[0, col] = 1; W[1, col] = 1  # C[0,0], C[1,0]
    col += 1
    
    # M5: (A11+A12), B22 -> -C11, C12
    U[0, col] = 1; U[1, col] = 1  # A[0,0], A[0,1]
    V[6, col] = 1  # B[1,1]
    W[0, col] = -1; W[4, col] = 1  # C[0,0], C[0,1]
    col += 1
    
    # M6: (A21-A11), (B11+B12) -> C22
    U[5, col] = 1; U[0, col] = -1  # A[1,0], A[0,0]
    V[0, col] = 1; V[1, col] = 1  # B[0,0], B[0,1]
    W[5, col] = 1  # C[1,1]
    col += 1
    
    # M7: (A12-A22), (B21+B22) -> C11, -C22
    U[1, col] = 1; U[6, col] = -1  # A[0,1], A[1,1]
    V[5, col] = 1; V[6, col] = 1  # B[1,0], B[1,1]
    W[0, col] = 1; W[5, col] = -1  # C[0,0], C[1,1]
    col += 1
    
    # A12*B21: 2x3x2, schoolbook rank 12
    # A12: rows [0,1], cols [2,3,4]
    # B21: rows [2,3,4], cols [0,1]
    # C11: rows [0,1], cols [0,1]
    for i in [0, 1]:
        for j in [2, 3, 4]:
            for k in [0, 1]:
                U[i*5+j, col] = 1
                V[j*5+k, col] = 1
                W[k*4+i, col] = 1
                col += 1
    
    # C12 = A11*B12 + A12*B22
    # A11*B12: 2x2x3, schoolbook rank 12
    # A11: rows [0,1], cols [0,1]
    # B12: rows [0,1], cols [2,3,4]
    # C12: rows [0,1], cols [2,3,4]
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [2, 3, 4]:
                U[i*5+j, col] = 1
                V[j*5+k, col] = 1
                W[k*4+i, col] = 1
                col += 1
    
    # A12*B22: 2x3x3, schoolbook rank 18
    # A12: rows [0,1], cols [2,3,4]
    # B22: rows [2,3,4], cols [2,3,4]
    # C12: rows [0,1], cols [2,3,4]
    for i in [0, 1]:
        for j in [2, 3, 4]:
            for k in [2, 3, 4]:
                U[i*5+j, col] = 1
                V[j*5+k, col] = 1
                W[k*4+i, col] = 1
                col += 1
    
    # C21 = A21*B11 + A22*B21
    # A21*B11: 2x2x2, use Strassen
    # A21: rows [2,3], cols [0,1]
    # B11: rows [0,1], cols [0,1]
    # C21: rows [2,3], cols [0,1]
    # U indices: A[2,0]->U[2*5+0]=U[10], A[2,1]->U[11], A[3,0]->U[15], A[3,1]->U[16]
    # V indices: B[0,0]->V[0], B[0,1]->V[1], B[1,0]->V[5], B[1,1]->V[6]
    # W indices: C[2,0]->W[0*4+2]=W[2], C[2,1]->W[1*4+2]=W[6], C[3,0]->W[0*4+3]=W[3], C[3,1]->W[1*4+3]=W[7]
    
    # M1: A21+A22 (local), B11+B22 (local) -> C21, C31 (local)
    U[10, col] = 1; U[16, col] = 1  # A[2,0], A[3,1]
    V[0, col] = 1; V[6, col] = 1  # B[0,0], B[1,1]
    W[2, col] = 1; W[7, col] = 1  # C[2,0], C[3,1]
    col += 1
    
    # M2: (A21+A22 local), B11 -> C21 local
    U[15, col] = 1; U[16, col] = 1  # A[3,0], A[3,1]
    V[0, col] = 1  # B[0,0]
    W[3, col] = 1  # C[3,0]
    col += 1
    
    # M3: A21 local, (B12-B22 local) -> C22 local, C32 local
    U[10, col] = 1  # A[2,0]
    V[1, col] = 1; V[6, col] = -1  # B[0,1], B[1,1]
    W[6, col] = 1; W[7, col] = 1  # C[2,1], C[3,1]
    col += 1
    
    # M4: A22 local, (B21-B11 local) -> C21 local, C31 local
    U[16, col] = 1  # A[3,1]
    V[5, col] = 1; V[0, col] = -1  # B[1,0], B[0,0]
    W[2, col] = 1; W[3, col] = 1  # C[2,0], C[3,0]
    col += 1
    
    # M5: (A21+A22 local), B22 local -> -C21 local, C22 local
    U[10, col] = 1; U[11, col] = 1  # A[2,0], A[2,1]
    V[6, col] = 1  # B[1,1]
    W[2, col] = -1; W[6, col] = 1  # C[2,0], C[2,1]
    col += 1
    
    # M6: (A21-A21 local), (B11+B12 local) -> C32 local
    U[15, col] = 1; U[10, col] = -1  # A[3,0], A[2,0]
    V[0, col] = 1; V[1, col] = 1  # B[0,0], B[0,1]
    W[7, col] = 1  # C[3,1]
    col += 1
    
    # M7: (A22-A22 local), (B21+B22 local) -> C21 local, -C32 local
    U[11, col] = 1; U[16, col] = -1  # A[2,1], A[3,1]
    V[5, col] = 1; V[6, col] = 1  # B[1,0], B[1,1]
    W[2, col] = 1; W[7, col] = -1  # C[2,0], C[3,1]
    col += 1
    
    # A22*B21: 2x3x2, schoolbook rank 12
    # A22: rows [2,3], cols [2,3,4]
    # B21: rows [2,3,4], cols [0,1]
    # C21: rows [2,3], cols [0,1]
    for i in [2, 3]:
        for j in [2, 3, 4]:
            for k in [0, 1]:
                U[i*5+j, col] = 1
                V[j*5+k, col] = 1
                W[k*4+i, col] = 1
                col += 1
    
    # C22 = A21*B12 + A22*B22
    # A21*B12: 2x2x3, schoolbook rank 12
    # A21: rows [2,3], cols [0,1]
    # B12: rows [0,1], cols [2,3,4]
    # C22: rows [2,3], cols [2,3,4]
    for i in [2, 3]:
        for j in [0, 1]:
            for k in [2, 3, 4]:
                U[i*5+j, col] = 1
                V[j*5+k, col] = 1
                W[k*4+i, col] = 1
                col += 1
    
    # A22*B22: 2x3x3, schoolbook rank 18
    # A22: rows [2,3], cols [2,3,4]
    # B22: rows [2,3,4], cols [2,3,4]
    # C22: rows [2,3], cols [2,3,4]
    for i in [2, 3]:
        for j in [2, 3, 4]:
            for k in [2, 3, 4]:
                U[i*5+j, col] = 1
                V[j*5+k, col] = 1
                W[k*4+i, col] = 1
                col += 1
    
    # Verify col == total_rank
    assert col == total_rank, f"Expected {total_rank} columns, got {col}"
    
    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, tol=1e-6):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c] for c in range(U_mat.shape[1]))
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fall back to schoolbook
        U_sb = np.zeros((n*m, n*m*p))
        V_sb = np.zeros((m*p, n*m*p))
        W_sb = np.zeros((p*n, n*m*p))
        col_sb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col_sb] = 1
                    V_sb[j*p+k, col_sb] = 1
                    W_sb[k*n+i, col_sb] = 1
                    col_sb += 1
        U, V, W = U_sb, V_sb, W_sb
    
    U = U.tolist()
    V = V.tolist()
    W = W.tolist()
    return {"U": U, "V": V, "W": W}
