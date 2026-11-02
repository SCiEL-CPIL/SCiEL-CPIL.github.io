"""Matrix multiplication 3x4x5 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]
DEADLINE_SECONDS = 140  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Build expected tensor
    expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute decomposed tensor
    D = np.einsum('xr,yr,zr->xyz', U_np, V_np, W_np)
    
    return np.allclose(D, expected, atol=tol)

def _get_strassen_2x2x2():
    """Returns list of 7 (u, v, w) vectors for 2x2x2 Strassen."""
    terms = []
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    terms.append(([1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]))
    terms.append(([0, 0, 0, 1], [0, 1, 0, 0], [-1, 0, 1, 0]))
    terms.append(([1, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]))
    terms.append(([ -1, 0, 1, 0], [1, 0, 0, 1], [0, 0, 0, -1]))
    terms.append(([0, 1, 0, -1], [0, 1, 0, 1], [0, 1, 0, -1]))
    return terms

def _get_rank_11_2x2x3():
    """
    Returns list of 11 (u, v, w) vectors for 2x2x3.
    Decomposes into 2x2x2 (rank 7) + 2x2x1 (rank 4).
    A is 2x2, B is 2x3. C is 2x3.
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
    """
    Returns list of 4 (u, v, w) vectors for 1x2x2.
    A is 1x2, B is 2x2. C is 1x2.
    """
    terms = []
    terms.append(([1, 0], [1, 0, 0, 0], [1, 0]))
    terms.append(([1, 0], [0, 1, 0, 0], [0, 1]))
    terms.append(([0, 1], [0, 0, 1, 0], [1, 0]))
    terms.append(([0, 1], [0, 0, 0, 1], [0, 1]))
    return terms

def _get_rank_6_1x2x3():
    """
    Returns list of 6 (u, v, w) vectors for 1x2x3.
    A is 1x2, B is 2x3. C is 1x3.
    """
    terms = []
    terms.append(([1, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0]))
    terms.append(([1, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0]))
    terms.append(([1, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1]))
    terms.append(([0, 1], [0, 0, 0, 1, 0, 0], [1, 0, 0]))
    terms.append(([0, 1], [0, 0, 0, 0, 1, 0], [0, 1, 0]))
    terms.append(([0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 1]))
    return terms

def _build_block_decomposition(n, m, p):
    """
    Build a block-based decomposition using Strassen and schoolbook sub-blocks.
    Returns (U, V, W, rank) or None if verification fails.
    """
    m1, m2 = 2, 2
    p1, p2 = 2, 3
    
    rank = 0
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_terms(u_local, v_local, w_local, u_map, v_map, w_map):
        nonlocal rank
        for u, v, w in zip(u_local, v_local, w_local):
            u_g = [0] * (n * m)
            v_g = [0] * (m * p)
            w_g = [0] * (p * n)
            
            for li, gi in u_map.items():
                u_g[gi] = u[li]
            for li, gi in v_map.items():
                v_g[gi] = v[li]
            for li, gi in w_map.items():
                w_g[gi] = w[li]
            
            U_cols.append(u_g)
            V_cols.append(v_g)
            W_cols.append(w_g)
            rank += 1
    
    s_terms = _get_strassen_2x2x2()
    u_map_1a = {0: 0, 1: 1, 2: 4, 3: 5}
    v_map_1a = {0: 0, 1: 1, 2: 5, 3: 6}
    w_map_1a = {0: 0, 1: 1, 2: 3, 3: 4}
    add_terms([u for u, _, _ in s_terms], [v for _, v, _ in s_terms], [w for _, _, w in s_terms], u_map_1a, v_map_1a, w_map_1a)
    
    u_map_1b = {0: 2, 1: 3, 2: 6, 3: 7}
    v_map_1b = {0: 10, 1: 11, 2: 15, 3: 16}
    w_map_1b = {0: 0, 1: 1, 2: 3, 3: 4}
    add_terms([u for u, _, _ in s_terms], [v for _, v, _ in s_terms], [w for _, _, w in s_terms], u_map_1b, v_map_1b, w_map_1b)
    
    r11_terms = _get_rank_11_2x2x3()
    u_map_2a = {0: 0, 1: 1, 2: 4, 3: 5}
    v_map_2a = {0: 2, 1: 3, 2: 4, 3: 7, 4: 8, 5: 9}
    w_map_2a = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7}
    add_terms([u for u, _, _ in r11_terms], [v for _, v, _ in r11_terms], [w for _, _, w in r11_terms], u_map_2a, v_map_2a, w_map_2a)
    
    u_map_2b = {0: 2, 1: 3, 2: 6, 3: 7}
    v_map_2b = {0: 12, 1: 13, 2: 14, 3: 17, 4: 18, 5: 19}
    w_map_2b = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7}
    add_terms([u for u, _, _ in r11_terms], [v for _, v, _ in r11_terms], [w for _, _, w in r11_terms], u_map_2b, v_map_2b, w_map_2b)
    
    r4_terms = _get_rank_4_1x2x2()
    u_map_3a = {0: 8, 1: 9}
    v_map_3a = {0: 0, 1: 1, 2: 5, 3: 6}
    w_map_3a = {0: 6, 1: 7}
    add_terms([u for u, _, _ in r4_terms], [v for _, v, _ in r4_terms], [w for _, _, w in r4_terms], u_map_3a, v_map_3a, w_map_3a)
    
    u_map_3b = {0: 10, 1: 11}
    v_map_3b = {0: 10, 1: 11, 2: 15, 3: 16}
    w_map_3b = {0: 6, 1: 7}
    add_terms([u for u, _, _ in r4_terms], [v for _, v, _ in r4_terms], [w for _, _, w in r4_terms], u_map_3b, v_map_3b, w_map_3b)
    
    r6_terms = _get_rank_6_1x2x3()
    u_map_4a = {0: 8, 1: 9}
    v_map_4a = {0: 2, 1: 3, 2: 4, 3: 7, 4: 8, 5: 9}
    w_map_4a = {0: 8, 1: 9, 2: 10}
    add_terms([u for u, _, _ in r6_terms], [v for _, v, _ in r6_terms], [w for _, _, w in r6_terms], u_map_4a, v_map_4a, w_map_4a)
    
    u_map_4b = {0: 10, 1: 11}
    v_map_4b = {0: 12, 1: 13, 2: 14, 3: 17, 4: 18, 5: 19}
    w_map_4b = {0: 8, 1: 9, 2: 10}
    add_terms([u for u, _, _ in r6_terms], [v for _, v, _ in r6_terms], [w for _, _, w in r6_terms], u_map_4b, v_map_4b, w_map_4b)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for i in range(n * m):
            U[i][c] = U_cols[c][i]
        for i in range(m * p):
            V[i][c] = V_cols[c][i]
        for i in range(p * n):
            W[i][c] = W_cols[c][i]
    
    return U, V, W, rank

def _build_schoolbook(n, m, p):
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
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    n, m, p = DIMENSIONS
    
    try:
        U, V, W, rank = _build_block_decomposition(n, m, p)
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    U, V, W, rank = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
