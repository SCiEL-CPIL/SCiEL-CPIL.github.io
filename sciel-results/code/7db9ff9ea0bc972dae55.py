"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

import numpy as np

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify if the tensor decomposition is exact."""
    if not U or not V or not W:
        return False
    
    rank = len(U[0])
    # Check shapes
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for i in range(rank):
        if len(U[i]) != rank or len(V[i]) != rank or len(W[i]) != rank:
            return False

    # Construct T
    T = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1

    # Compute T_approx
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Transpose to (rank, dim) for easier einsum?
    # T_approx[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
    # U_np is (nm, rank)
    # V_np is (mp, rank)
    # W_np is (pn, rank)
    
    # Efficient computation:
    # For each r, outer product
    # Or use einsum: 'ur,vr,wr->uvw'
    T_approx = np.einsum('ur,vr,wr->uvw', U_np, V_np, W_np)
    
    return np.allclose(T_approx, T)

def _construct_strassen_238():
    """
    Constructs rank 44 decomposition for 2x3x8 matrix multiplication.
    Uses 4 blocks of 2x3x2.
    Each 2x3x2 block uses Strassen (rank 7) for 2x2x2 part and Outer Product (rank 4) for 2x1x2 part.
    Total rank: 4 * (7 + 4) = 44.
    """
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(col, u_idx, u_coeff, v_idx, v_coeff, w_idx, w_coeff):
        for idx, c in zip(u_idx, u_coeff):
            U[idx][col] += c
        for idx, c in zip(v_idx, v_coeff):
            V[idx][col] += c
        for idx, c in zip(w_idx, w_coeff):
            W[idx][col] += c

    current_col = 0
    
    for k_block in range(4):
        k0 = k_block * 2
        k1 = k_block * 2 + 1
        
        # Indices for A (2x3)
        a_idx = 0
        b_idx = 1
        c_idx = 3
        d_idx = 4
        e_idx_a = 2 # A[0,2]
        f_idx_a = 5 # A[1,2]
        
        # Indices for B (3x8)
        # B[j, k] index is j*8 + k
        e_idx = 0 * p + k0
        f_idx = 0 * p + k1
        g_idx = 1 * p + k0
        h_idx = 1 * p + k1
        
        e2_idx = 2 * p + k0
        f2_idx = 2 * p + k1
        
        # Indices for W (8x2) -> C^T
        # C[i, k] index in W is k*2 + i
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        # Part 1: 2x2x2 Strassen (A[0:2, 0:2] * B[0:2, k0:k1])
        # M1 = (a+d)(e+h)
        add_term(current_col, [a_idx, d_idx], [1, 1], [e_idx, h_idx], [1, 1], [w_00, w_11], [1, 1])
        current_col += 1
        
        # M2 = (c+d)e
        add_term(current_col, [c_idx, d_idx], [1, 1], [e_idx], [1], [w_10, w_11], [1, -1])
        current_col += 1
        
        # M3 = a(f-h)
        add_term(current_col, [a_idx], [1], [f_idx, h_idx], [1, -1], [w_01, w_11], [1, 1])
        current_col += 1
        
        # M4 = d(g-e)
        add_term(current_col, [d_idx], [1], [g_idx, e_idx], [1, -1], [w_00, w_10], [1, 1])
        current_col += 1
        
        # M5 = (a+b)h
        add_term(current_col, [a_idx, b_idx], [1, 1], [h_idx], [1], [w_00, w_01], [-1, 1])
        current_col += 1
        
        # M6 = (c-a)(e+f)
        add_term(current_col, [c_idx, a_idx], [1, -1], [e_idx, f_idx], [1, 1], [w_11], [1])
        current_col += 1
        
        # M7 = (b-d)(g+h)
        add_term(current_col, [b_idx, d_idx], [1, -1], [g_idx, h_idx], [1, 1], [w_00], [1])
        current_col += 1
        
        # Part 2: 2x1x2 Outer Product (A[0:2, 2] * B[2, k0:k1])
        # C[0, k0] = A[0,2] * B[2, k0]
        add_term(current_col, [e_idx_a], [1], [e2_idx], [1], [w_00], [1])
        current_col += 1
        
        # C[0, k1] = A[0,2] * B[2, k1]
        add_term(current_col, [e_idx_a], [1], [f2_idx], [1], [w_01], [1])
        current_col += 1
        
        # C[1, k0] = A[1,2] * B[2, k0]
        add_term(current_col, [f_idx_a], [1], [e2_idx], [1], [w_10], [1])
        current_col += 1
        
        # C[1, k1] = A[1,2] * B[2, k1]
        add_term(current_col, [f_idx_a], [1], [f2_idx], [1], [w_11], [1])
        current_col += 1

    return {"U": U, "V": V, "W": W}

def _construct_schoolbook():
    """
    Constructs rank 48 schoolbook decomposition.
    """
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

def construct():
    import time
    start_time = time.time()
    deadline = 120.0
    
    # Try Strassen-based rank 44
    try:
        cand1 = _construct_strassen_238()
        if time.time() < deadline - 5:
            if _verify_decomposition(cand1["U"], cand1["V"], cand1["W"], 2, 3, 8):
                return cand1
    except Exception:
        pass
        
    # Fallback to Schoolbook rank 48
    try:
        cand2 = _construct_schoolbook()
        if time.time() < deadline - 1:
            if _verify_decomposition(cand2["U"], cand2["V"], cand2["W"], 2, 3, 8):
                return cand2
    except Exception:
        pass
        
    # Return schoolbook regardless if verification fails due to bug or timeout
    return _construct_schoolbook()
# EVOLVE-BLOCK-END
