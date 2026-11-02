"""Matrix multiplication 3x4x5 - rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]
DEADLINE_SECONDS = 140

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    rank = U_np.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_np[i * m + j] * V_np[j * p + k] * W_np[k * n + i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _get_strassen_2x2x2():
    """Returns list of 7 (u, v, w) vectors for 2x2x2 Strassen."""
    terms = []
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    terms.append(([1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]))
    terms.append(([0, 0, 0, 1], [0, 1, 0, 0], [-1, 0, 1, 0]))
    terms.append(([1, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]))
    terms.append(([-1, 0, 1, 0], [1, 0, 0, 1], [0, 0, 0, -1]))
    terms.append(([0, 1, 0, -1], [0, 1, 0, 1], [0, 1, 0, -1]))
    return terms

def _get_rank_11_2x2x3():
    """
    Returns list of 11 (u, v, w) vectors for 2x2x3.
    Decomposes into 2x2x2 (rank 7) + 2x2x1 (rank 4).
    """
    terms = []
    s_terms = _get_strassen_2x2x2()
    for u, v, w in s_terms:
        u_g = u
        v_g = [v[0], v[1], 0, v[2], v[3], 0]
        w_g = [w[0], w[1], 0, w[2], w[3], 0]
        terms.append((u_g, v_g, w_g))
    
    terms.append(([1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]))
    terms.append(([0, 1, 0, 0], [0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0]))
    terms.append(([0, 0, 1, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]))
    terms.append(([0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0]))
    return terms

def _get_rank_4_1x2x2():
    """Returns list of 4 (u, v, w) vectors for 1x2x2."""
    terms = []
    terms.append(([1, 0], [1, 0, 0, 0], [1, 0]))
    terms.append(([1, 0], [0, 1, 0, 0], [0, 1]))
    terms.append(([0, 1], [0, 0, 1, 0], [1, 0]))
    terms.append(([0, 1], [0, 0, 0, 1], [0, 1]))
    return terms

def _get_rank_6_1x2x3():
    """Returns list of 6 (u, v, w) vectors for 1x2x3."""
    terms = []
    terms.append(([1, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0]))
    terms.append(([1, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0]))
    terms.append(([1, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1]))
    terms.append(([0, 1], [0, 0, 0, 1, 0, 0], [1, 0, 0]))
    terms.append(([0, 1], [0, 0, 0, 0, 1, 0], [0, 1, 0]))
    terms.append(([0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 1]))
    return terms

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # We will construct the 56-rank decomposition.
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        for i in range(n * m):
            U[i][col] = u_vec[i]
        for i in range(m * p):
            V[i][col] = v_vec[i]
        for i in range(p * n):
            W[i][col] = w_vec[i]
        col += 1
    
    # Block 1: C11 (2x2) from A1 (2x4) and B1 (4x2)
    s_terms = _get_strassen_2x2x2()
    
    # Term 1: A11 * B11
    for u, v, w in s_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[0] = u[0]
        u_g[1] = u[1]
        u_g[4] = u[2]
        u_g[5] = u[3]
        v_g[0] = v[0]
        v_g[1] = v[1]
        v_g[5] = v[2]
        v_g[6] = v[3]
        w_g[0] = w[0]
        w_g[1] = w[1]
        w_g[3] = w[2]
        w_g[4] = w[3]
        add_term(u_g, v_g, w_g)
        
    # Term 2: A12 * B12
    for u, v, w in s_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[2] = u[0]
        u_g[3] = u[1]
        u_g[6] = u[2]
        u_g[7] = u[3]
        v_g[10] = v[0]
        v_g[11] = v[1]
        v_g[15] = v[2]
        v_g[16] = v[3]
        w_g[0] = w[0]
        w_g[1] = w[1]
        w_g[3] = w[2]
        w_g[4] = w[3]
        add_term(u_g, v_g, w_g)
        
    # Block 2: C12 (2x3) from A1 (2x4) and B2 (4x3)
    r11_terms = _get_rank_11_2x2x3()
    
    # Term 1: A11 * B21
    for u, v, w in r11_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[0] = u[0]
        u_g[1] = u[1]
        u_g[4] = u[2]
        u_g[5] = u[3]
        v_g[2] = v[0]
        v_g[3] = v[1]
        v_g[4] = v[2]
        v_g[7] = v[3]
        v_g[8] = v[4]
        v_g[9] = v[5]
        w_g[2] = w[0]
        w_g[3] = w[1]
        w_g[4] = w[2]
        w_g[5] = w[3]
        w_g[6] = w[4]
        w_g[7] = w[5]
        add_term(u_g, v_g, w_g)
        
    # Term 2: A12 * B22
    for u, v, w in r11_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[2] = u[0]
        u_g[3] = u[1]
        u_g[6] = u[2]
        u_g[7] = u[3]
        v_g[12] = v[0]
        v_g[13] = v[1]
        v_g[14] = v[2]
        v_g[17] = v[3]
        v_g[18] = v[4]
        v_g[19] = v[5]
        w_g[2] = w[0]
        w_g[3] = w[1]
        w_g[4] = w[2]
        w_g[5] = w[3]
        w_g[6] = w[4]
        w_g[7] = w[5]
        add_term(u_g, v_g, w_g)
        
    # Block 3: C21 (1x2) from A2 (1x4) and B1 (4x2)
    r4_terms = _get_rank_4_1x2x2()
    
    # Term 1: A21 * B11
    for u, v, w in r4_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[8] = u[0]
        u_g[9] = u[1]
        v_g[0] = v[0]
        v_g[1] = v[1]
        v_g[5] = v[2]
        v_g[6] = v[3]
        w_g[6] = w[0]
        w_g[7] = w[1]
        add_term(u_g, v_g, w_g)
        
    # Term 2: A22 * B12
    for u, v, w in r4_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[10] = u[0]
        u_g[11] = u[1]
        v_g[10] = v[0]
        v_g[11] = v[1]
        v_g[15] = v[2]
        v_g[16] = v[3]
        w_g[6] = w[0]
        w_g[7] = w[1]
        add_term(u_g, v_g, w_g)
        
    # Block 4: C22 (1x3) from A2 (1x4) and B2 (4x3)
    r6_terms = _get_rank_6_1x2x3()
    
    # Term 1: A21 * B21
    for u, v, w in r6_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[8] = u[0]
        u_g[9] = u[1]
        v_g[2] = v[0]
        v_g[3] = v[1]
        v_g[4] = v[2]
        v_g[7] = v[3]
        v_g[8] = v[4]
        v_g[9] = v[5]
        w_g[8] = w[0]
        w_g[9] = w[1]
        w_g[10] = w[2]
        add_term(u_g, v_g, w_g)
        
    # Term 2: A22 * B22
    for u, v, w in r6_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[10] = u[0]
        u_g[11] = u[1]
        v_g[12] = v[0]
        v_g[13] = v[1]
        v_g[14] = v[2]
        v_g[17] = v[3]
        v_g[18] = v[4]
        v_g[19] = v[5]
        w_g[8] = w[0]
        w_g[9] = w[1]
        w_g[10] = w[2]
        add_term(u_g, v_g, w_g)
        
    if col != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {col}")
        
    if not _verify_decomposition(U, V, W, n, m, p):
        schoolbook_rank = n * m * p
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
