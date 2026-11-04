import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    T[u,v,w] = sum_r U[u,r]*V[v,r]*W[w,r] should equal 1
    iff u=i*m+k, v=k*p+j, w=j*n+i with consistent i,k,j.
    """
    rank = U.shape[1]
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u = i * m + k
                v = k * p + j
                w = j * n + i
                T_target[u, v, w] = 1.0
    
    T_decomp = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_decomp += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
    
    error = np.max(np.abs(T_decomp - T_target))
    return error < tol, error

def _strassen_block(U, V, W, col, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, m, p, n):
    """
    Adds 7 rank-1 terms for a Strassen decomposition of a 2x2x2 block.
    """
    u_a = a_rows[0] * m + a_cols[0]
    u_b = a_rows[0] * m + a_cols[1]
    u_c = a_rows[1] * m + a_cols[0]
    u_d = a_rows[1] * m + a_cols[1]
    
    v_e = b_rows[0] * p + b_cols[0]
    v_f = b_rows[0] * p + b_cols[1]
    v_g = b_rows[1] * p + b_cols[0]
    v_h = b_rows[1] * p + b_cols[1]
    
    w_00 = c_cols[0] * n + c_rows[0]
    w_01 = c_cols[1] * n + c_rows[0]
    w_10 = c_cols[0] * n + c_rows[1]
    w_11 = c_cols[1] * n + c_rows[1]
    
    # M1: (a+d)(e+h)
    U[u_a, col] += 1.0
    U[u_d, col] += 1.0
    V[v_e, col] += 1.0
    V[v_h, col] += 1.0
    W[w_00, col] += 1.0
    W[w_11, col] += 1.0
    col += 1
    
    # M2: (c+d)e
    U[u_c, col] += 1.0
    U[u_d, col] += 1.0
    V[v_e, col] += 1.0
    W[w_10, col] += 1.0
    W[w_11, col] += -1.0
    col += 1
    
    # M3: a(f-h)
    U[u_a, col] += 1.0
    V[v_f, col] += 1.0
    V[v_h, col] += -1.0
    W[w_01, col] += 1.0
    W[w_11, col] += 1.0
    col += 1
    
    # M4: d(g-e)
    U[u_d, col] += 1.0
    V[v_g, col] += 1.0
    V[v_e, col] += -1.0
    W[w_00, col] += 1.0
    W[w_10, col] += 1.0
    col += 1
    
    # M5: (a+b)h
    U[u_a, col] += 1.0
    U[u_b, col] += 1.0
    V[v_h, col] += 1.0
    W[w_00, col] += -1.0
    W[w_01, col] += 1.0
    col += 1
    
    # M6: (c-a)(e+f)
    U[u_c, col] += 1.0
    U[u_a, col] += -1.0
    V[v_e, col] += 1.0
    V[v_f, col] += 1.0
    W[w_11, col] += 1.0
    col += 1
    
    # M7: (b-d)(g+h)
    U[u_b, col] += 1.0
    U[u_d, col] += -1.0
    V[v_g, col] += 1.0
    V[v_h, col] += 1.0
    W[w_00, col] += 1.0
    col += 1
    
    return col

def _construct_rank_88():
    """
    Construct the rank-88 decomposition using Strassen blocks.
    C = A * B where A is 3x4, B is 4x8, C is 3x8.
    
    Split A into A_top (2x4) and A_bottom (1x4).
    Split B into B_left (4x4) and B_right (4x4).
    
    For A_top * B: split into 4 Strassen 2x2x2 blocks, each handling 2 columns of B.
    For A_bottom * B: schoolbook 32 terms.
    
    Total: 4*7 + 4*7 + 32 = 88.
    """
    n, m, p = DIMENSIONS  # 3, 4, 8
    rank = 88
    U = np.zeros((n * m, rank), dtype=float)  # (12, 88)
    V = np.zeros((m * p, rank), dtype=float)  # (32, 88)
    W = np.zeros((p * n, rank), dtype=float)  # (24, 88)
    
    col = 0
    
    # Part 1a: A[0:2, 0:2] * B[0:2, :]
    for k_start in range(0, 8, 2):
        col = _strassen_block(
            U, V, W, col,
            a_rows=[0, 1], a_cols=[0, 1],
            b_rows=[0, 1], b_cols=[k_start, k_start+1],
            c_rows=[0, 1], c_cols=[k_start, k_start+1],
            m=m, p=p, n=n
        )
    
    # Part 1b: A[0:2, 2:4] * B[2:4, :]
    for k_start in range(0, 8, 2):
        col = _strassen_block(
            U, V, W, col,
            a_rows=[0, 1], a_cols=[2, 3],
            b_rows=[2, 3], b_cols=[k_start, k_start+1],
            c_rows=[0, 1], c_cols=[k_start, k_start+1],
            m=m, p=p, n=n
        )
    
    # Part 2: A[2:3, :] * B[:, :]
    for k in range(4):
        for j in range(8):
            U[2 * m + k, col] = 1.0
            V[k * p + j, col] = 1.0
            W[j * n + 2, col] = 1.0
            col += 1
    
    if col != rank:
        raise Exception(f"Rank mismatch: expected {rank}, got {col}")
    
    valid, error = _verify_decomposition(U, V, W, n, m, p)
    if not valid:
        raise Exception(f"Verification failed with error {error}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    result = _construct_rank_88()
    
    if time.time() > deadline:
        # Should not happen, but safety guard
        pass
    
    return result
