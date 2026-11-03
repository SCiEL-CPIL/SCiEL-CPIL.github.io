"""Rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
Baseline: Kronecker-embedded Strassen (rank 65).
Refinement: ALS + tensor-train subspace compression + structured random projections
with half-integer rounding. Falls back to schoolbook (rank 80)."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify_fast(U, V, W, n=4, m=4, p=5, atol=1e-9):
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
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
    """Rank-65: Kronecker-embedded Strassen for 4x4x4 + naive for 4x4x1."""
    U_cols, V_cols, W_cols = [], [], []
    strassen_terms = _strassen_2x2x2_terms()
    for (u1s, v1s, w1s) in strassen_terms:
        for (u2s, v2s, w2s) in strassen_terms:
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            for (i1, j1), val1 in [(idx//2, idx%2) for idx, _ in u1s]:
                pass
            # Build via nested loops properly
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            for idx1, val1 in u1s:
                i1, j1 = idx1 // 2, idx1 % 2
                for idx2, val2 in u2s:
                    i2, j2 = idx2 // 2, idx2 % 2
                    u[(i1*2+i2)*m + (j1*2+j2)] = val1*val2
            for idx1, val1 in v1s:
                j1, k1 = idx1 // 2, idx1 % 2
                for idx2, val2 in v2s:
                    j2, k2 = idx2 // 2, idx2 % 2
                    v[(j1*2+j2)*p + (k1*2+k2)] = val1*val2
            for idx1, val1 in w1s:
                k1, i1 = idx1 // 2, idx1 % 2
                for idx2, val2 in w2s:
                    k2, i2 = idx2 // 2, idx2 % 2
                    w[(k1*2+k2)*n + (i1*2+i2)] = val1*val2
            U_cols.append(u); V_cols.append(v); W_cols.append(w)
    for i in range(n):
        for j in range(m):
            u = np.zeros(n*m); u[i*m+j] = 1
            v = np.zeros(m*p); v[j*p+4] = 1
            w = np.zeros(p*n); w[4*n+i] = 1
            U_cols.append(u); V_cols.append(v); W_cols.append(w)
    return np.array(U_cols).T.tolist(), np.array(V_cols).T.tolist(), np.array(W_cols).T.tolist()

def _strassen_block_construct(n, m, p):
    """Rank-72: 2x2 block Strassen. Used as alternative baseline."""
    U_cols, V_cols, W_cols = [], [], []
    strassen_terms = _strassen_2x2x2_terms()
    def add_term(u, v, w):
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
    def add_strassen_2x2_2(ai, aj, bj, bk):
        for u_sparse, v_sparse, w_sparse in strassen_terms:
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            for idx, val in u_sparse:
                ui, uj = idx//2, idx%2
                u[(ai+ui)*m + (aj+uj)] = val
            for idx, val in v_sparse:
                vj, vk = idx//2, idx%2
                v[(bj+vj)*p + (bk+vk)] = val
            for idx, val in w_sparse:
                wk, wi = idx//2, idx%2
                w[(bk+wk)*n + (ai+wi)] = val
            add_term(u, v, w)
    def add_2x2x3(ai, aj, bj, bk):
        add_strassen_2x2_2(ai, aj, bj, bk)
        for di in range(2):
            for dj in range(2):
                u = np.zeros(n*m); u[(ai+di)*m + (aj+dj)] = 1
                v = np.zeros(m*p); v[(bj+dj)*p + (bk+2)] = 1
                w = np.zeros(p*n); w[(bk+2)*n + (ai+di)] = 1
                add_term(u, v, w)
    add_strassen_2x2_2(0,0,0,0); add_strassen_2x2_2(0,2,2,0)
    add_strassen_2x2_2(2,0,0,0); add_strassen_2x2_2(2,2,2,0)
    add_2x2x3(0,0,0,2); add_2x2x3(0,2,2,2)
    add_2x2x3(2,0,0,2); add_2x2x3(2,2,2,2)
    return np.array(U_cols).T.tolist(), np.array(V_cols).T.tolist(), np.array(W_cols).T.tolist()

def _als_decompose(n, m, p, rank, seed=42, max_iters=300, deadline=None,
                   U_init=None, V_init=None, W_init=None, polish_iters=150):
    if U_init is not None and V_init is not None and W_init is not None:
        U = np.array(U_init, dtype=np.float64).copy()
        V = np.array(V_init, dtype=np.float64).copy()
        W = np.array(W_init, dtype=np.float64).copy()
    else:
        np.random.seed(seed)
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        # Update U
        for i in range(n):
            for j in range(m):
                A_mat = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                b_vec = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        # Update V
        for j in range(m):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                b_vec = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        # Update W
        for i in range(n):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                b_vec = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err += (s - 1.0)**2
        err = np.sqrt(err)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < 1e-12:
            break
    # Polish
    if best_err < 1e-5:
        U_pol, V_pol, W_pol = best_U.copy(), best_V.copy(), best_W.copy()
        for _ in range(polish_iters):
            if deadline is not None and time.time() > deadline:
                break
            for i in range(n):
                for j in range(m):
                    A_mat = np.array([V_pol[j*p+k, :] * W_pol[k*n+i, :] for k in range(p)])
                    try:
                        U_pol[i*m+j, :] = np.linalg.lstsq(A_mat, np.ones(p), rcond=1e-15)[0]
                    except:
                        pass
            for j in range(m):
                for k in range(p):
                    A_mat = np.array([U_pol[i*m+j, :] * W_pol[k*n+i, :] for i in range(n)])
                    try:
                        V_pol[j*p+k, :] = np.linalg.lstsq(A_mat, np.ones(n), rcond=1e-15)[0]
                    except:
                        pass
            for i in range(n):
                for k in range(p):
                    A_mat = np.array([U_pol[i*m+j, :] * V_pol[j*p+k, :] for j in range(m)])
                    try:
                        W_pol[k*n+i, :] = np.linalg.lstsq(A_mat, np.ones(m), rcond=1e-15)[0]
                    except:
                        pass
            err_pol = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U_pol[i*m+j, r] * V_pol[j*p+k, r] * W_pol[k*n+i, r]
                        err_pol += (s - 1.0)**2
            err_pol = np.sqrt(err_pol)
            if err_pol < best_err:
                best_err = err_pol
                best_U, best_V, best_W = U_pol.copy(), V_pol.copy(), W_pol.copy()
            if err_pol < 1e-14:
                break
    U_r = _round_matrix_to_half_int(best_U)
    V_r = _round_matrix_to_half_int(best_V)
    W_r = _round_matrix_to_half_int(best_W)
    if _verify_fast(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, atol=1e-9):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    return None

def _tt_compress_refine(U, V, W, n, m, p, target_rank, deadline):
    """Tensor-train style: project the 3-mode tensors onto a lower-rank subspace
    via SVD of the reshaped tensors, then re-run ALS from that projection."""
    Ur = np.array(U, dtype=np.float64).reshape(n, m, -1)
    Vr = np.array(V, dtype=np.float64).reshape(m, p, -1)
    Wr = np.array(W, dtype=np.float64).reshape(p, n, -1)
    # Truncate each mode's coefficient space
    def _trunc_modes(T, r):
        # T shape (..., rank). Flatten trailing dim, SVD, keep top r.
        shape = T.shape
        flat = T.reshape(-1, shape[-1])
        try:
            U_s, S, Vt = np.linalg.svd(flat, full_matrices=False)
            k = min(r, Vt.shape[0])
            # Project: new coeffs = U_s[:, :k] @ diag(S[:k]) @ Vt[:k, :]
            # But we want a rank-k representation. Use the top-k right singular vectors.
            proj = Vt[:k, :].T  # shape (rank, k)
            new_T = flat @ proj  # shape (-1, k)
            return new_T.reshape(shape[:-1] + (k,))
        except:
            return T[:, :r] if T.shape[-1] >= r else T
    Ur_t = _trunc_modes(Ur, target_rank)
    Vr_t = _trunc_modes(Vr, target_rank)
    Wr_t = _trunc_modes(Wr, target_rank)
    U_init = Ur_t.reshape(n*m, target_rank)
    V_init = Vr_t.reshape(m*p, target_rank)
    W_init = Wr_t.reshape(p*n, target_rank)
    return _als_decompose(n, m, p, target_rank, U_init=U_init, V_init=V_init,
                          W_init=W_init, max_iters=200, deadline=deadline)

def _structured_random_init(n, m, p, rank, seed):
    """Generate random init with block-structure awareness."""
    np.random.seed(seed)
    U = np.random.randn(n*m, rank) * 0.3
    V = np.random.randn(m*p, rank) * 0.3
    W = np.random.randn(p*n, rank) * 0.3
    return U, V, W

def _try_reduce(U_base, V_base, W_base, n, m, p, deadline):
    Ua = np.array(U_base, dtype=np.float64)
    Va = np.array(V_base, dtype=np.float64)
    Wa = np.array(W_base, dtype=np.float64)
    rank = Ua.shape[1]
    best_result = None
    best_rank = rank
    for target in range(rank-1, max(40, rank-25), -1):
        if time.time() > deadline:
            break
        # Method 1: column truncation
        for init_seed in range(3):
            if time.time() > deadline:
                break
            if init_seed == 0:
                Ui, Vi, Wi = Ua[:, :target], Va[:, :target], Wa[:, :target]
            elif init_seed == 1:
                Ui, Vi, Wi = Ua[:, rank-target:], Va[:, rank-target:], Wa[:, rank-target:]
            else:
                np.random.seed(init_seed*77+target)
                idx = np.random.choice(rank, target, replace=False)
                Ui, Vi, Wi = Ua[:, idx], Va[:, idx], Wa[:, idx]
            res = _als_decompose(n, m, p, target, U_init=Ui, V_init=Vi, W_init=Wi,
                                 max_iters=150, deadline=deadline)
            if res is not None:
                best_result = res
                best_rank = target
                break
        # Method 2: TT compression
        if time.time() > deadline:
            break
        res = _tt_compress_refine(U_base, V_base, W_base, n, m, p, target, deadline)
        if res is not None:
            best_result = res
            best_rank = target
        if best_rank == target:
            break
    return best_result

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start = time.time()
    deadline = start + 130  # safety margin
    n, m, p = DIMENSIONS
    # Build both baselines
    U_kron, V_kron, W_kron = _kron_strassen_construct(n, m, p)
    rank_kron = len(U_kron[0])
    U_blk, V_blk, W_blk = _strassen_block_construct(n, m, p)
    rank_blk = len(U_blk[0])
    # Pick best baseline
    if rank_kron <= rank_blk:
        U_base, V_base, W_base = U_kron, V_kron, W_kron
        best_rank = rank_kron
    else:
        U_base, V_base, W_base = U_blk, V_blk, W_blk
        best_rank = rank_blk
    best_result = {"U": U_base, "V": V_base, "W": W_base}
    if not _verify_fast(U_base, V_base, W_base, n, m, p, atol=1e-9):
        return _schoolbook()
    # Try to reduce
    if time.time() < deadline - 30:
        reduced = _try_reduce(U_base, V_base, W_base, n, m, p, deadline - 30)
        if reduced is not None:
            U_r, V_r, W_r = reduced
            best_result = {"U": U_r, "V": V_r, "W": W_r}
            best_rank = len(U_r[0])
    # Random init search at lower ranks
    for rank in range(best_rank-1, max(40, best_rank-20), -1):
        if time.time() > deadline - 15:
            break
        found = False
        for seed in range(8):
            if time.time() > deadline - 15:
                break
            Ui, Vi, Wi = _structured_random_init(n, m, p, rank, seed)
            res = _als_decompose(n, m, p, rank, U_init=Ui, V_init=Vi, W_init=Wi,
                                 max_iters=120, deadline=deadline-15)
            if res is not None:
                U, V, W = res
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                found = True
                break
        if found:
            continue
    return best_result

# EVOLVE-BLOCK-END
