import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _strassen_2x2():
    """Returns U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W = np.array([
        [1, 0, 0, 1, -1, 0, -1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    return U, V, W

def _kron_product(A, B):
    """Kronecker product of two matrices A and B using numpy."""
    return np.kron(A, B)

def _schoolbook_4x4x1():
    """Schoolbook decomposition for 4x4x1 (rank 16)."""
    n, m, p = 4, 4, 1
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Reshape U to (n, m, rank)
    U_reshaped = U.reshape(n, m, rank)
    # Reshape V to (m, p, rank)
    V_reshaped = V.reshape(m, p, rank)
    # Reshape W to (p, n, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # Compute tensor product sum
    # T[i,j,k] = sum_c U_reshaped[i,j,c] * V_reshaped[j,k,c] * W_reshaped[k,i,c]
    
    # Use einsum for efficiency
    # U_reshaped: (n, m, r)
    # V_reshaped: (m, p, r)
    # W_reshaped: (p, n, r)
    
    # We want to compute sum over c.
    # This is a contraction.
    # T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    # Expected is 1.0 for all i,j,k
    return np.allclose(T, 1.0, atol=1e-10)

def _schoolbook():
    """Schoolbook decomposition of rank n*m*p = 144."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # 1. Get Strassen 2x2x2 factors
    U2, V2, W2 = _strassen_2x2()

    # 2. Kronecker product for 4x4x4
    U4_44 = _kron_product(U2, U2)
    V4_44 = _kron_product(V2, V2)
    W4_44 = _kron_product(W2, W2)

    # 3. Schoolbook for 4x4x1
    U4_16, V4_16, W4_16 = _schoolbook_4x4x1()

    # 4. Assemble for 4x4x9
    # Rank = 49 + 49 + 16 = 114
    total_rank = 114

    U_full = np.zeros((16, total_rank))
    V_full = np.zeros((36, total_rank))
    W_full = np.zeros((36, total_rank))

    # Part 1: 4x4x4 (cols 0-3 of B, rows 0-3 of C)
    # U_full[:, 0:49] = U4_44
    U_full[:, 0:49] = U4_44
    
    # V mapping: r_v (0..15) -> j = r_v // 4, k = r_v % 4. Global row = j*9 + k.
    # V4_44 is 16x49.
    # We need to place V4_44 into V_full.
    # V_full rows are indexed by (j, k) where j in 0..3, k in 0..8.
    # For Part 1, k is 0..3.
    # V4_44 row r corresponds to (j, k).
    # V_full row j*9+k gets V4_44 row r.
    for r_v in range(16):
        j = r_v // 4
        k = r_v % 4
        global_v_row = j * 9 + k
        V_full[global_v_row, 0:49] = V4_44[r_v, :]
        
    # W mapping: r_w (0..15) -> k = r_w // 4, i = r_w % 4. Global row = k*4 + i.
    for r_w in range(16):
        k = r_w // 4
        i = r_w % 4
        global_w_row = k * 4 + i
        W_full[global_w_row, 0:49] = W4_44[r_w, :]

    # Part 2: 4x4x4 (cols 4-7 of B, rows 4-7 of C)
    offset = 49
    U_full[:, offset:offset+49] = U4_44
    
    for r_v in range(16):
        j = r_v // 4
        k_local = r_v % 4
        k_global = k_local + 4
        global_v_row = j * 9 + k_global
        V_full[global_v_row, offset:offset+49] = V4_44[r_v, :]
        
    for r_w in range(16):
        k_local = r_w // 4
        i = r_w % 4
        k_global = k_local + 4
        global_w_row = k_global * 4 + i
        W_full[global_w_row, offset:offset+49] = W4_44[r_w, :]

    # Part 3: 4x4x1 (col 8 of B, row 8 of C)
    offset = 98
    U_full[:, offset:offset+16] = U4_16
    
    # V4_16 is 4x16 (rows 0..3 correspond to j=0..3, k=0).
    # Global V row for j, k=8 is j*9 + 8.
    for r_v in range(4):
        global_v_row = r_v * 9 + 8
        V_full[global_v_row, offset:offset+16] = V4_16[r_v, :]
        
    # W4_16 is 4x16 (rows 0..3 correspond to k=0, i=0..3).
    # Global W row for k=8, i is 8*4 + i.
    for r_w in range(4):
        global_w_row = 8 * 4 + r_w
        W_full[global_w_row, offset:offset+16] = W4_16[r_w, :]

    # Verify
    if time.time() > deadline:
        # If we are running out of time, return the best we have (which should be valid)
        # But verification might be slow? No, einsum is fast.
        pass

    if not _verify_decomposition(U_full, V_full, W_full, 4, 4, 9):
        return _schoolbook()
        
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
