import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # We use the rank 55 decomposition:
    # Split B (3x10) into 5 blocks of (3x2).
    # For each block, we compute C_block (2x2) = A (2x3) * B_block (3x2).
    # We split A into A_12 (2x2, cols 0,1) and A_3 (2x1, col 2).
    # We split B_block into B_12 (2x2, rows 0,1) and B_3 (1x2, row 2).
    # C_block = A_12 * B_12 + A_3 * B_3.
    # Rank(A_12 * B_12) = Rank(M_2,2,2) = 7 (Strassen).
    # Rank(A_3 * B_3) = Rank(M_2,1,2) = 4 (Schoolbook).
    # Total per block = 11.
    # Total rank = 5 * 11 = 55.
    
    rank = 55
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col_idx = 0
    
    # Helper to index U, V, W
    # U[i, j] corresponds to A[i, j] -> index i*m + j
    # V[j, k] corresponds to B[j, k] -> index j*p + k
    # W[k, i] corresponds to C[i, k] -> index k*n + i (Trace convention W[k*n+i, r])
    
    # Strassen formulas for 2x2 * 2x2
    # Inputs: a00, a01, a10, a11 and b00, b01, b10, b11
    # Outputs: c00, c01, c10, c11
    
    def add_strassen_term(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        nonlocal col_idx
        if time.time() > deadline:
            return False
            
        # M1 = (a00 + a11) * (b00 + b11)
        U[a00, col_idx] += 1.0
        U[a11, col_idx] += 1.0
        V[b00, col_idx] += 1.0
        V[b11, col_idx] += 1.0
        W[c00, col_idx] += 1.0
        W[c11, col_idx] += 1.0
        col_idx += 1
        
        # M2 = (a10 + a11) * b00
        U[a10, col_idx] += 1.0
        U[a11, col_idx] += 1.0
        V[b00, col_idx] += 1.0
        W[c10, col_idx] += 1.0
        W[c11, col_idx] -= 1.0
        col_idx += 1
        
        # M3 = a00 * (b01 - b11)
        U[a00, col_idx] += 1.0
        V[b01, col_idx] += 1.0
        V[b11, col_idx] -= 1.0
        W[c01, col_idx] += 1.0
        W[c11, col_idx] += 1.0
        col_idx += 1
        
        # M4 = a11 * (b10 - b00)
        U[a11, col_idx] += 1.0
        V[b10, col_idx] += 1.0
        V[b00, col_idx] -= 1.0
        W[c00, col_idx] += 1.0
        W[c10, col_idx] += 1.0
        col_idx += 1
        
        # M5 = (a00 + a01) * b11
        U[a00, col_idx] += 1.0
        U[a01, col_idx] += 1.0
        V[b11, col_idx] += 1.0
        W[c00, col_idx] -= 1.0
        W[c01, col_idx] += 1.0
        col_idx += 1
        
        # M6 = (a10 - a00) * (b00 + b01)
        U[a10, col_idx] += 1.0
        U[a00, col_idx] -= 1.0
        V[b00, col_idx] += 1.0
        V[b01, col_idx] += 1.0
        W[c11, col_idx] += 1.0
        col_idx += 1
        
        # M7 = (a01 - a11) * (b10 + b11)
        U[a01, col_idx] += 1.0
        U[a11, col_idx] -= 1.0
        V[b10, col_idx] += 1.0
        V[b11, col_idx] += 1.0
        W[c00, col_idx] += 1.0
        col_idx += 1
        
        return True

    def add_schoolbook_term(a_i, b_j, c_ik):
        nonlocal col_idx
        if time.time() > deadline:
            return False
        U[a_i, col_idx] += 1.0
        V[b_j, col_idx] += 1.0
        W[c_ik, col_idx] += 1.0
        col_idx += 1
        return True

    # Iterate over 5 blocks of 2 columns
    for b_idx in range(5):
        k0 = 2 * b_idx
        k1 = 2 * b_idx + 1
        
        # A indices (shared across blocks)
        # A is 2x3. Rows 0,1. Cols 0,1,2.
        a00 = 0 * m + 0
        a01 = 0 * m + 1
        a10 = 1 * m + 0
        a11 = 1 * m + 1
        a02 = 0 * m + 2
        a12 = 1 * m + 2
        
        # B indices for this block
        # B is 3x10. Rows 0,1,2. Cols k0, k1.
        b00 = 0 * p + k0
        b01 = 0 * p + k1
        b10 = 1 * p + k0
        b11 = 1 * p + k1
        b20 = 2 * p + k0
        b21 = 2 * p + k1
        
        # C indices for this block
        # C is 2x10. Rows 0,1. Cols k0, k1.
        # W index is k*n + i
        c00 = k0 * n + 0
        c01 = k1 * n + 0
        c10 = k0 * n + 1
        c11 = k1 * n + 1
        
        # Part 1: A_12 * B_12 (7 terms)
        if not add_strassen_term(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
            break
            
        # Part 2: A_3 * B_3 (4 terms)
        # C_0k0 += A_02 * B_20
        # C_0k1 += A_02 * B_21
        # C_1k0 += A_12 * B_20
        # C_1k1 += A_12 * B_21
        
        if not add_schoolbook_term(a02, b20, c00): break
        if not add_schoolbook_term(a02, b21, c01): break
        if not add_schoolbook_term(a12, b20, c10): break
        if not add_schoolbook_term(a12, b21, c11): break

    # Verification
    def verify(U_mat, V_mat, W_mat, r):
        n_v, m_v, p_v = 2, 3, 10
        for i in range(n_v):
            for j in range(m_v):
                for k in range(p_v):
                    s = 0.0
                    for c in range(r):
                        s += U_mat[i * m_v + j, c] * V_mat[j * p_v + k, c] * W_mat[k * n_v + i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W, rank):
        # Fallback to schoolbook rank 60
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb), dtype=float)
        V_sb = np.zeros((m * p, rank_sb), dtype=float)
        W_sb = np.zeros((p * n, rank_sb), dtype=float)
        col_sb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col_sb] = 1
                    V_sb[j * p + k, col_sb] = 1
                    W_sb[k * n + i, col_sb] = 1
                    col_sb += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
