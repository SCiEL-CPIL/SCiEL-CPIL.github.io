import numpy as np
import time

DIMENSIONS = (2, 4, 4)

def _get_strassen_factors():
    """
    Returns U, V, W for 2x2x2 Strassen multiplication.
    U, V, W are (4, 7) matrices.
    Indexing:
    U: A00, A01, A10, A11
    V: B00, B01, B10, B11
    W: C00, C01, C10, C11
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _verify(U, V, W):
    """Check exact decomposition on random integer matrices."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    rng = np.random.RandomState(42)
    for trial in range(3):
        A = rng.randint(-5, 6, (n, m)).astype(float)
        B = rng.randint(-5, 6, (m, p)).astype(float)
        C = A @ B
        
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        M1 = np.einsum('ijc,jkc->ikc', U_r, V_r)
        C_calc = np.einsum('ikc,kic->ik', M1, W_r)
        
        if not np.allclose(C, C_calc, atol=1e-6):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    # Rank 28
    rank = 28
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U2, V2, W2 = _get_strassen_factors()
    
    # Mapping of 2x2 indices to flattened indices
    # A (n x m): i in 0..1, j in 0..3
    # A_L (i, 0..1): idx = i*4 + j
    # A_R (i, 2..3): idx = i*4 + j
    
    # B (m x p): j in 0..3, k in 0..3
    # B_T (0..1, k): idx = j*4 + k
    # B_B (2..3, k): idx = j*4 + k
    
    # C (n x p): i in 0..1, k in 0..3
    # W idx = k*n + i = k*2 + i
    
    # Instance 1: A_L * B_T1 (k=0,1)
    # A_L rows in U:
    # A_00 -> 0, A_01 -> 1, A_10 -> 4, A_11 -> 5
    u_rows_1 = [0, 1, 4, 5]
    # B_T1 rows in V:
    # B_00 -> 0, B_01 -> 1, B_10 -> 4, B_11 -> 5
    v_rows_1 = [0, 1, 4, 5]
    # W rows for C[:, 0:2]:
    # C_00 -> 0, C_10 -> 1, C_01 -> 2, C_11 -> 3
    w_rows_1 = [0, 1, 2, 3]
    
    for c in range(7):
        for idx_u, val in zip(u_rows_1, U2[:, c]):
            U[idx_u, c] = val
        for idx_v, val in zip(v_rows_1, V2[:, c]):
            V[idx_v, c] = val
        for idx_w, val in zip(w_rows_1, W2[:, c]):
            W[idx_w, c] = val
            
    # Instance 2: A_L * B_T2 (k=2,3)
    # A_L same
    u_rows_2 = [0, 1, 4, 5]
    # B_T2 rows in V:
    # B_02 -> 2, B_03 -> 3, B_12 -> 6, B_13 -> 7
    v_rows_2 = [2, 3, 6, 7]
    # W rows for C[:, 2:4]:
    # C_02 -> 4, C_12 -> 5, C_03 -> 6, C_13 -> 7
    w_rows_2 = [4, 5, 6, 7]
    
    for c in range(7):
        col_offset = 7 + c
        for idx_u, val in zip(u_rows_2, U2[:, c]):
            U[idx_u, col_offset] = val
        for idx_v, val in zip(v_rows_2, V2[:, c]):
            V[idx_v, col_offset] = val
        for idx_w, val in zip(w_rows_2, W2[:, c]):
            W[idx_w, col_offset] = val
            
    # Instance 3: A_R * B_B1 (k=0,1)
    # A_R rows in U:
    # A_02 -> 2, A_03 -> 3, A_12 -> 6, A_13 -> 7
    u_rows_3 = [2, 3, 6, 7]
    # B_B1 rows in V:
    # B_20 -> 8, B_21 -> 9, B_30 -> 12, B_31 -> 13
    v_rows_3 = [8, 9, 12, 13]
    # W rows for C[:, 0:2]:
    w_rows_3 = [0, 1, 2, 3]
    
    for c in range(7):
        col_offset = 14 + c
        for idx_u, val in zip(u_rows_3, U2[:, c]):
            U[idx_u, col_offset] = val
        for idx_v, val in zip(v_rows_3, V2[:, c]):
            V[idx_v, col_offset] = val
        for idx_w, val in zip(w_rows_3, W2[:, c]):
            W[idx_w, col_offset] = val
            
    # Instance 4: A_R * B_B2 (k=2,3)
    # A_R same
    u_rows_4 = [2, 3, 6, 7]
    # B_B2 rows in V:
    # B_22 -> 10, B_23 -> 11, B_32 -> 14, B_33 -> 15
    v_rows_4 = [10, 11, 14, 15]
    # W rows for C[:, 2:4]:
    w_rows_4 = [4, 5, 6, 7]
    
    for c in range(7):
        col_offset = 21 + c
        for idx_u, val in zip(u_rows_4, U2[:, c]):
            U[idx_u, col_offset] = val
        for idx_v, val in zip(v_rows_4, V2[:, c]):
            V[idx_v, col_offset] = val
        for idx_w, val in zip(w_rows_4, W2[:, c]):
            W[idx_w, col_offset] = val
            
    if not _verify(U, V, W):
        # Fallback to naive if verification fails (should not happen)
        U_naive = np.zeros((n*m, n*m*p))
        V_naive = np.zeros((m*p, n*m*p))
        W_naive = np.zeros((p*n, n*m*p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i*m*p + j*p + k
                    U[i*m+j, idx] = 1
                    V[j*p+k, idx] = 1
                    W[k*n+i, idx] = 1
        return {'U': U_naive, 'V': V_naive, 'W': W_naive}
        
    return {'U': U, 'V': V, 'W': W}
