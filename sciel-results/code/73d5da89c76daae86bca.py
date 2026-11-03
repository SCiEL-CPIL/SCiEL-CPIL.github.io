"""Search for a rank-minimal exact tensor decomposition of 4x4 * 4x5 matrix multiplication.
Uses Kronecker-embedded Strassen (rank 65) and Block Strassen (rank 72) as baselines.
Attempts rank reduction via vectorized ALS with half-integer rounding and structured column removal.
Includes a polishing phase to ensure exactness.
Falls back to schoolbook (rank 80) if all else fails."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify_fast(U, V, W, n=4, m=4, p=5, atol=1e-9):
    """Fast verification using numpy with float64."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    if Ua.ndim == 1:
        Ua = Ua.reshape(-1, 1)
    if Va.ndim == 1:
        Va = Va.reshape(-1, 1)
    if Wa.ndim == 1:
        Wa = Wa.reshape(-1, 1)
        
    rank = Ua.shape[1]
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = Ur[:, :, c]
        vc = Vr[:, :, c]
        wc = Wr[:, :, c]
        wcT = wc.T
        for j in range(m):
            T[:, j, :] = T[:, j, :] + np.outer(uc[:, j], vc[j, :]) * wcT
    return np.allclose(T, 1.0, atol=atol)

def _round_to_half_integer(x):
    return round(x * 2) / 2.0

def _round_matrix_to_half_int(M):
    return np.array([[_round_to_half_integer(x) for x in row] for row in M])

def _strassen_2x2x2_terms():
    """Return the 7 Strassen 2x2x2 terms as sparse (index, value) lists for U, V, W."""
    return [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]

def _kron_strassen_construct(n, m, p):
    """Construct a rank-65 decomposition: 4x4x4 via Kronecker Strassen (49) + 4x4x1 naive (16)."""
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    strassen_terms = _strassen_2x2x2_terms()
    
    for (u1_sparse, v1_sparse, w1_sparse) in strassen_terms:
        for (u2_sparse, v2_sparse, w2_sparse) in strassen_terms:
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            
            for (idx1, val1) in u1_sparse:
                i1, j1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in u2_sparse:
                    i2, j2 = idx2 // 2, idx2 % 2
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    u[i * m + j] = val1 * val2
            
            for (idx1, val1) in v1_sparse:
                j1, k1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in v2_sparse:
                    j2, k2 = idx2 // 2, idx2 % 2
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    v[j * p + k] = val1 * val2
            
            for (idx1, val1) in w1_sparse:
                k1, i1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in w2_sparse:
                    k2, i2 = idx2 // 2, idx2 % 2
                    k = k1 * 2 + k2
                    i = i1 * 2 + i2
                    w[k * n + i] = val1 * val2
            
            add_term(u, v, w)
    
    for i in range(n):
        for j in range(m):
            u = np.zeros(n*m); u[i * m + j] = 1
            v = np.zeros(m*p); v[j * p + 4] = 1
            w = np.zeros(p*n); w[4 * n + i] = 1
            add_term(u, v, w)
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()

def _strassen_block_construct(n, m, p):
    """Construct a rank-64 decomposition using 2x2 blocks.
    Splits B into B_left (4x2) and B_right (4x3).
    B_right is split into B_mid (4x2) and B_last (4x1).
    
    Part 1: A * B_left (4x2). Rank 28.
    Part 2: A * B_mid (4x2). Rank 28.
    Part 3: A * B_last (4x1). Rank 8.
    Total Rank: 64.
    """
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
        
    def get_u_idx(gi, gj):
        return gi * m + gj
        
    def get_v_idx(gj, gk):
        return gj * p + gk
        
    def get_w_idx(gk, gi):
        return gk * n + gi

    strassen_terms = _strassen_2x2x2_terms()

    def add_strassen_2x2_2(ai, aj, bj, bk):
        """Add 7 Strassen terms for A[ai:ai+2, aj:aj+2] @ B[bj:bj+2, bk:bk+2] -> C[ai:ai+2, bk:bk+2]"""
        for u_sparse, v_sparse, w_sparse in strassen_terms:
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            for idx, val in u_sparse:
                ui, uj = idx // 2, idx % 2
                u[get_u_idx(ai+ui, aj+uj)] = val
            for idx, val in v_sparse:
                vj, vk = idx // 2, idx % 2
                v[get_v_idx(bj+vj, bk+vk)] = val
            for idx, val in w_sparse:
                wk, wi = idx // 2, idx % 2
                w[get_w_idx(bk+wk, ai+wi)] = val
            add_term(u, v, w)

    def add_naive_2x2_1(ai, aj, bj, bk):
        """Add 4 Naive terms for A[ai:ai+2, aj:aj+2] @ B[bj:bj+2, bk:bk+1] -> C[ai:ai+2, bk:bk+1]"""
        for di in range(2):
            for dj in range(2):
                u = np.zeros(n*m); u[get_u_idx(ai+di, aj+dj)] = 1
                v = np.zeros(m*p); v[get_v_idx(bj+dj, bk)] = 1
                w = np.zeros(p*n); w[get_w_idx(bk, ai+di)] = 1
                add_term(u, v, w)

    def add_4x4x2(ai, aj, bj, bk):
        """Add 28 terms for A[ai:ai+4, aj:aj+4] @ B[bj:bj+4, bk:bk+2] -> C[ai:ai+4, bk:bk+2]"""
        # C_top = A00 B00 + A01 B10
        add_strassen_2x2_2(ai, aj, bj, bk)       # A00 * B00
        add_strassen_2x2_2(ai, aj+2, bj+2, bk)   # A01 * B10
        
        # C_bot = A10 B00 + A11 B10
        add_strassen_2x2_2(ai+2, aj, bj, bk)     # A10 * B00
        add_strassen_2x2_2(ai+2, aj+2, bj+2, bk) # A11 * B10

    def add_4x4x1(ai, aj, bj, bk):
        """Add 8 terms for A[ai:ai+4, aj:aj+4] @ B[bj:bj+4, bk:bk+1] -> C[ai:ai+4, bk:bk+1]"""
        # C_top = A00 B00 + A01 B10 (where B is 2x1)
        add_naive_2x2_1(ai, aj, bj, bk)       # A00 * B00
        add_naive_2x2_1(ai, aj+2, bj+2, bk)   # A01 * B10
        
        # C_bot = A10 B00 + A11 B10
        add_naive_2x2_1(ai+2, aj, bj, bk)     # A10 * B00
        add_naive_2x2_1(ai+2, aj+2, bj+2, bk) # A11 * B10

    # Part 1: C[:, 0:2] = A * B[:, 0:2]
    add_4x4x2(0, 0, 0, 0)
    
    # Part 2: C[:, 2:4] = A * B[:, 2:4]
    add_4x4x2(0, 0, 0, 2)
    
    # Part 3: C[:, 4:5] = A * B[:, 4:5]
    add_4x4x1(0, 0, 0, 4)

    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()

def _compute_error(U, V, W, n, m, p):
    """Compute the decomposition error efficiently."""
    rank = U.shape[1]
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = Ur[:, :, c]
        vc = Vr[:, :, c]
        wc = Wr[:, :, c]
        wcT = wc.T
        for j in range(m):
            T[:, j, :] = T[:, j, :] + np.outer(uc[:, j], vc[j, :]) * wcT
    return np.sqrt(np.sum((T - 1.0) ** 2))

def _als_decompose_vectorized(n, m, p, rank, seed=42, max_iters=300, deadline=None,
                               U_init=None, V_init=None, W_init=None, polish_iters=100):
    """Vectorized ALS with rounding to half-integers."""
    if U_init is not None and V_init is not None and W_init is not None:
        U = np.array(U_init, dtype=np.float64).copy()
        V = np.array(V_init, dtype=np.float64).copy()
        W = np.array(W_init, dtype=np.float64).copy()
    else:
        np.random.seed(seed)
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    def _als_iteration():
        # Optimize U
        for i in range(n):
            for j in range(m):
                V_slice = V[j*p:(j+1)*p, :]
                W_slice = np.array([W[k*n+i, :] for k in range(p)])
                A_mat = V_slice * W_slice
                b_vec = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Optimize V
        for j in range(m):
            for k in range(p):
                U_slice = np.array([U[i*m+j, :] for i in range(n)])
                W_slice = W[k*n:(k+1)*n, :]
                A_mat = U_slice * W_slice
                b_vec = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Optimize W
        for i in range(n):
            for k in range(p):
                U_slice = U[i*m:(i+1)*m, :]
                V_slice = np.array([V[j*p+k, :] for j in range(m)])
                A_mat = U_slice * V_slice
                b_vec = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        _als_iteration()
        err = _compute_error(U, V, W, n, m, p)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-12:
            break
    
    for it in range(polish_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        _als_iteration()
        err = _compute_error(U, V, W, n, m, p)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-14:
            break
    
    U_r = _round_matrix_to_half_int(best_U)
    V_r = _round_matrix_to_half_int(best_V)
    W_r = _round_matrix_to_half_int(best_W)
    
    if _verify_fast(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, atol=1e-9):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    return None

def _try_reduce_rank(U_base, V_base, W_base, n, m, p, target_rank, deadline):
    """Try to reduce the rank to target_rank from the given baseline."""
    Ua = np.array(U_base, dtype=np.float64)
    Va = np.array(V_base, dtype=np.float64)
    Wa = np.array(W_base, dtype=np.float64)
    current_rank = Ua.shape[1]
    
    if current_rank <= target_rank:
        return (Ua.tolist(), Va.tolist(), Wa.tolist())
    
    # Strategy 1: Drop random columns and re-optimize
    for seed in range(10):
        if time.time() > deadline:
            break
        np.random.seed(seed)
        idx = np.random.choice(current_rank, target_rank, replace=False)
        U_init = Ua[:, idx]
        V_init = Va[:, idx]
        W_init = Wa[:, idx]
        
        result = _als_decompose_vectorized(n, m, p, target_rank, 
                                U_init=U_init, V_init=V_init, W_init=W_init,
                                max_iters=100, deadline=deadline, polish_iters=50)
        if result is not None:
            return result
            
    # Strategy 2: Random restart
    for seed in range(5):
        if time.time() > deadline:
            break
        result = _als_decompose_vectorized(n, m, p, target_rank, seed=seed, max_iters=100, 
                                deadline=deadline, polish_iters=50)
        if result is not None:
            return result
            
    return None

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Build baselines
    U_block, V_block, W_block = _strassen_block_construct(n, m, p)
    block_ok = _verify_fast(U_block, V_block, W_block, n, m, p, atol=1e-9)
    
    U_kron, V_kron, W_kron = _kron_strassen_construct(n, m, p)
    kron_ok = _verify_fast(U_kron, V_kron, W_kron, n, m, p, atol=1e-9)
    
    best_result = None
    best_rank = 80 # Schoolbook
    
    if block_ok:
        best_result = {"U": U_block, "V": V_block, "W": W_block}
        best_rank = len(U_block[0])
        
    if kron_ok and len(U_kron[0]) < best_rank:
        best_result = {"U": U_kron, "V": V_kron, "W": W_kron}
        best_rank = len(U_kron[0])
        
    if not block_ok and not kron_ok:
        return _schoolbook()
        
    # Use the best baseline for reduction
    if block_ok and (not kron_ok or len(U_block[0]) <= len(U_kron[0])):
        U_base, V_base, W_base = U_block, V_block, W_block
    else:
        U_base, V_base, W_base = U_kron, V_kron, W_kron
        
    # Try to reduce rank
    for target_rank in range(best_rank - 1, max(50, best_rank - 15), -1):
        if time.time() > deadline - 10:
            break
            
        reduced = _try_reduce_rank(U_base, V_base, W_base, n, m, p, target_rank, deadline - 10)
        if reduced is not None:
            U_red, V_red, W_red = reduced
            best_result = {"U": U_red, "V": V_red, "W": W_red}
            best_rank = target_rank
            # Update baseline for next iteration
            U_base, V_base, W_base = U_red, V_red, W_red
            
    return best_result

# EVOLVE-BLOCK-END
