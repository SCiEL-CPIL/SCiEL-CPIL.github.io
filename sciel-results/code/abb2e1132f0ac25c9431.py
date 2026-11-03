"""Search for a rank-minimal exact tensor decomposition of 4x4 * 4x5 matrix multiplication.
Uses Kronecker-embedded Strassen baseline (rank 65) with refined ALS optimization
and structural pruning to find lower-rank half-integer decompositions."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _verify_fast(U, V, W, n=4, m=4, p=5, atol=1e-9):
    """Fast verification using numpy with float64."""
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

def _compute_error(U, V, W, n=4, m=4, p=5):
    """Compute the L2 error of the decomposition."""
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
    return np.sqrt(np.sum((T - 1.0) ** 2))

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

def _als_decompose(n, m, p, rank, seed=42, max_iters=500, deadline=None,
                   U_init=None, V_init=None, W_init=None, polish_iters=100):
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
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        for i in range(n):
            for j in range(m):
                A_mat = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                b_vec = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        for j in range(m):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                b_vec = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        for i in range(n):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                b_vec = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        err = _compute_error(U.tolist(), V.tolist(), W.tolist(), n, m, p)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-12:
            break
    
    for it in range(polish_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        for i in range(n):
            for j in range(m):
                A_mat = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                b_vec = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        for j in range(m):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                b_vec = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        for i in range(n):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                b_vec = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        err = _compute_error(U.tolist(), V.tolist(), W.tolist(), n, m, p)
        
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

def _structural_prune(U_base, V_base, W_base, n, m, p, deadline):
    """Try to find a lower-rank decomposition by identifying and removing
    redundant or dependent columns from the baseline."""
    Ua = np.array(U_base, dtype=np.float64)
    Va = np.array(V_base, dtype=np.float64)
    Wa = np.array(W_base, dtype=np.float64)
    rank = Ua.shape[1]
    
    best_result = None
    best_rank = rank
    
    # Try to remove columns one at a time, starting from the end
    current_U = Ua.copy()
    current_V = Va.copy()
    current_W = Wa.copy()
    current_rank = rank
    
    removed = set()
    
    for trial in range(30):
        if time.time() > deadline:
            break
        
        # Try removing each column (in order of smallest norm)
        col_norms = np.linalg.norm(current_U, axis=0)
        order = np.argsort(col_norms)
        
        found = False
        for idx in order:
            if idx in removed:
                continue
            
            # Try removing this column
            keep_mask = np.ones(current_rank, dtype=bool)
            keep_mask[idx] = False
            
            U_test = current_U[:, keep_mask]
            V_test = current_V[:, keep_mask]
            W_test = current_W[:, keep_mask]
            
            # Try to recover with ALS
            result = _als_decompose(n, m, p, current_rank - 1,
                                   U_init=U_test, V_init=V_test, W_test=W_test,
                                   max_iters=100, deadline=deadline, polish_iters=30)
            
            if result is not None:
                U_test_r, V_test_r, W_test_r = result
                # Verify
                if _verify_fast(U_test_r, V_test_r, W_test_r, n, m, p, atol=1e-9):
                    current_U = np.array(U_test_r, dtype=np.float64)
                    current_V = np.array(V_test_r, dtype=np.float64)
                    current_W = np.array(W_test_r, dtype=np.float64)
                    current_rank -= 1
                    removed.add(idx)
                    found = True
                    if best_result is None or current_rank < best_rank:
                        best_result = (U_test_r, V_test_r, W_test_r)
                        best_rank = current_rank
                    break
        
        if not found:
            break
    
    return best_result

def _try_reduce_from_baseline(U_base, V_base, W_base, n, m, p, deadline):
    Ua = np.array(U_base, dtype=np.float64)
    Va = np.array(V_base, dtype=np.float64)
    Wa = np.array(W_base, dtype=np.float64)
    rank = Ua.shape[1]
    
    best_result = None
    best_rank = rank
    
    # Try reducing rank by dropping columns and re-optimizing
    for target_rank in range(rank - 1, max(45, rank - 20), -1):
        if time.time() > deadline:
            break
        
        for init_seed in range(5):
            if time.time() > deadline:
                break
            
            if init_seed == 0:
                U_init = Ua[:, :target_rank]
                V_init = Va[:, :target_rank]
                W_init = Wa[:, :target_rank]
            elif init_seed == 1:
                U_init = Ua[:, rank-target_rank:]
                V_init = Va[:, rank-target_rank:]
                W_init = Wa[:, rank-target_rank:]
            elif init_seed == 2:
                np.random.seed(init_seed * 100 + target_rank)
                idx = np.random.choice(rank, target_rank, replace=False)
                U_init = Ua[:, idx]
                V_init = Va[:, idx]
                W_init = Wa[:, idx]
            elif init_seed == 3:
                # Keep the Strassen part, drop some naive columns
                # The first 49 columns are Strassen, last 16 are naive
                if target_rank >= 49:
                    U_init = np.hstack([Ua[:, :49], Ua[:, 49:49+(target_rank-49)]])
                    V_init = np.hstack([Va[:, :49], Va[:, 49:49+(target_rank-49)]])
                    W_init = np.hstack([Wa[:, :49], Wa[:, 49:49+(target_rank-49)]])
                else:
                    continue
            else:
                np.random.seed(999 + target_rank)
                idx = np.random.choice(rank, target_rank, replace=False)
                U_init = Ua[:, idx] + np.random.randn(n*m, target_rank) * 0.05
                V_init = Va[:, idx] + np.random.randn(m*p, target_rank) * 0.05
                W_init = Wa[:, idx] + np.random.randn(p*n, target_rank) * 0.05
            
            result = _als_decompose(n, m, p, target_rank,
                                    U_init=U_init, V_init=V_init, W_init=W_init,
                                    max_iters=300, deadline=deadline, polish_iters=80)
            if result is not None:
                best_result = result
                best_rank = target_rank
                break
    
    return best_result

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
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    U_kron, V_kron, W_kron = _kron_strassen_construct(n, m, p)
    
    kron_ok = _verify_fast(U_kron, V_kron, W_kron, n, m, p, atol=1e-9)
    
    if not kron_ok:
        return _schoolbook()
    
    U_base, V_base, W_base = U_kron, V_kron, W_kron
    
    best_result = {"U": U_base, "V": V_base, "W": W_base}
    best_rank = len(U_base[0])
    
    # Try structural pruning first (faster, more targeted)
    if time.time() < deadline - 50:
        pruned = _structural_prune(U_base, V_base, W_base, n, m, p, deadline - 50)
        if pruned is not None:
            U_p, V_p, W_p = pruned
            if len(U_p[0]) < best_rank:
                best_result = {"U": U_p, "V": V_p, "W": W_p}
                best_rank = len(U_p[0])
                U_base, V_base, W_base = U_p, V_p, W_p
    
    # Try to reduce rank from the (possibly pruned) baseline
    if time.time() < deadline - 30:
        reduced = _try_reduce_from_baseline(U_base, V_base, W_base, n, m, p, deadline - 30)
        if reduced is not None:
            U_red, V_red, W_red = reduced
            if len(U_red[0]) < best_rank:
                best_result = {"U": U_red, "V": V_red, "W": W_red}
                best_rank = len(U_red[0])
    
    # Try random initializations at various ranks
    for rank in range(best_rank - 1, max(50, best_rank - 15), -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        for seed in range(10):
            if time.time() > deadline - 15:
                break
            
            result = _als_decompose(n, m, p, rank, seed=seed, max_iters=500,
                                    deadline=deadline - 15, polish_iters=100)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                found = True
                break
        
        if found:
            continue
    
    return best_result
# EVOLVE-BLOCK-END
