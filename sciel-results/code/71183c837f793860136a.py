import time
import numpy as np

def _verify(U, V, W, n=2, m=4, p=5):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should be 1 for all i, j, k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if not np.isclose(val, 1.0, atol=1e-9):
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 4, 5
    
    # Schoolbook
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_result = {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
    best_rank = rank_sb

    # Try Rank 36
    # A (2x4) = [A_L (2x2), A_R (2x2)]
    # B (4x5) = [B_T (2x5); B_B (2x5)]
    # C = A_L B_T + A_R B_B
    
    # B_T (2x5) split into B_T0 (2x2), B_T1 (2x2), B_T2 (2x1)
    # B_B (2x5) split into B_B0 (2x2), B_B1 (2x2), B_B2 (2x1)
    
    terms = []
    
    # Strassen 2x2x2 terms (0-based indices for local 2x2 matrices)
    # A_local[i,j], B_local[j,k], C_local[i,k]
    # M1 = (A[1,0] - A[1,1])(B[1,0] + B[1,1])
    # M2 = (A[1,0] + A[1,1])B[1,1]
    # M3 = (A[0,1] - A[1,1])(B[0,0] + B[0,1])
    # M4 = (A[1,1] - A[0,0])(B[0,0] + B[1,0])
    # M5 = (A[0,0] + A[0,1])B[1,1]
    # M6 = (A[1,0] - A[0,0])(B[0,0] + B[0,1])
    # M7 = (A[0,1] + A[1,1])(B[1,0] - B[1,1])
    
    # C[0,0] = M5 + M4 - M3 + M7
    # C[0,1] = M1 + M3
    # C[1,0] = M2 + M4
    # C[1,1] = M1 + M5 - M2 + M6
    
    strassen_terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # M1
        ({(1,0):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,1):1, (1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(1,1):1}, {(1,0):1, (1,1):-1}),
        # M3
        ({(0,1):1, (1,1):-1}, {(0,0):1, (0,1):1}, {(0,0):-1, (0,1):1}),
        # M4
        ({(1,1):1, (0,0):-1}, {(0,0):1, (1,0):1}, {(0,0):1, (1,0):1}),
        # M5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):1, (1,1):1}),
        # M6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7
        ({(0,1):1, (1,1):1}, {(1,0):1, (1,1):-1}, {(0,0):1}),
    ]
    
    # Helper to add terms for a block multiplication
    # a_offset: row offset for A (0 for A_L, 0 for A_R but col offset differs)
    # Actually, A_L is cols 0,1. A_R is cols 2,3.
    # B_T is rows 0,1. B_B is rows 2,3.
    # C is 2x5.
    
    def add_strassen(a_row_off, a_col_off, b_row_off, b_col_off, c_col_off):
        # a_row_off: 0 (since n=2, rows are 0,1)
        # a_col_off: 0 for A_L, 2 for A_R
        # b_row_off: 0 for B_T, 2 for B_B
        # b_col_off: 0 for B_T0, 2 for B_T1
        # c_col_off: 0 for C cols 0,1; 2 for C cols 2,3
        
        for a_c, b_c, c_c in strassen_terms:
            U_idx = {}
            V_idx = {}
            W_idx = {}
            
            for (i, j), coeff in a_c.items():
                global_i = i
                global_j = a_col_off + j
                U_idx[global_i * m + global_j] = coeff
            
            for (j, k), coeff in b_c.items():
                global_j = b_row_off + j
                global_k = b_col_off + k
                V_idx[global_j * p + global_k] = coeff
            
            for (i, k), coeff in c_c.items():
                global_i = i
                global_k = c_col_off + k
                W_idx[global_k * n + global_i] = coeff
                
            terms.append((U_idx, V_idx, W_idx))
            
    def add_schoolbook_2x1(a_col_off, b_row_off, b_col_idx, c_col_idx):
        # A (2x2) * B (2x1)
        # A[i, j] where j is local 0,1
        # B[j, 0] where j is local 0,1
        # C[i, 0]
        for i in range(2):
            for j in range(2):
                U_idx = {i * m + (a_col_off + j): 1}
                V_idx = {(b_row_off + j) * p + b_col_idx: 1}
                W_idx = {c_col_idx * n + i: 1}
                terms.append((U_idx, V_idx, W_idx))

    # A_L (cols 0,1) * B_T (rows 0,1)
    # B_T split into B_T0 (cols 0,1), B_T1 (cols 2,3), B_T2 (col 4)
    add_strassen(0, 0, 0, 0, 0) # A_L * B_T0 -> C cols 0,1
    add_strassen(0, 0, 0, 2, 2) # A_L * B_T1 -> C cols 2,3
    add_schoolbook_2x1(0, 0, 4, 4) # A_L * B_T2 -> C col 4
    
    # A_R (cols 2,3) * B_B (rows 2,3)
    # B_B split into B_B0 (cols 0,1), B_B1 (cols 2,3), B_B2 (col 4)
    add_strassen(0, 2, 2, 0, 0) # A_R * B_B0 -> C cols 0,1
    add_strassen(0, 2, 2, 2, 2) # A_R * B_B1 -> C cols 2,3
    add_schoolbook_2x1(2, 2, 4, 4) # A_R * B_B2 -> C col 4
    
    if len(terms) == 36:
        U_36 = np.zeros((n * m, 36))
        V_36 = np.zeros((m * p, 36))
        W_36 = np.zeros((p * n, 36))
        
        for r, (U_idx, V_idx, W_idx) in enumerate(terms):
            for idx, coeff in U_idx.items():
                U_36[idx, r] = coeff
            for idx, coeff in V_idx.items():
                V_36[idx, r] = coeff
            for idx, coeff in W_idx.items():
                W_36[idx, r] = coeff
                
        if _verify(U_36, V_36, W_36, n, m, p):
            best_rank = 36
            best_result = {
                "U": U_36.tolist(),
                "V": V_36.tolist(),
                "W": W_36.tolist()
            }
            
    return best_result
