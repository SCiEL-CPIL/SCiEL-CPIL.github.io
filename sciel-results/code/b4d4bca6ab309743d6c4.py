"""Search for a rank-minimal exact tensor decomposition of 4x4 * 4x5 matrix multiplication.
Uses Kronecker-embedded Strassen (rank 65) as baseline, then attempts refinement via
improved ALS with multi-strategy rounding and high-precision polish.
Falls back to schoolbook (rank 80) if all else fails."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _verify_fast(U, V, W, n=4, m=4, p=5, atol=1e-9):
    """Fast verification using numpy with float64, checking tolerance."""
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

def _compute_error(U, V, W, n, m, p, rank):
    """Compute the L2 error of the decomposition efficiently."""
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
    return np.sqrt(np.sum((T - 1.0)**2))

def _round_to_half_integer(x):
    """Round x to the nearest half-integer."""
    return round(x * 2) / 2.0

def _round_matrix_to_half_int(M):
    """Round all entries of matrix M to nearest half-integer."""
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
    """Construct a rank-65 decomposition using Kronecker product embedding.
    4x4x4 via Kronecker-embedded Strassen (rank 49) + 4x4x1 naive (rank 16)."""
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

def _als_decompose(n, m, p, rank, seed=42, max_iters=300, deadline=None,
                   U_init=None, V_init=None, W_init=None, polish_iters=200):
    """Try to find a rank-r decomposition using alternating least squares with rounding.
    Returns (U, V, W) as lists if found, else None."""
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
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        _als_iteration()
        err = _compute_error(U, V, W, n, m, p, rank)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-12:
            break
    
    # Polish phase with tighter convergence
    for it in range(polish_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        _als_iteration()
        err = _compute_error(U, V, W, n, m, p, rank)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-14:
            break
    
    # Try multiple rounding strategies
    candidates = []
    
    # Strategy 1: Round to nearest half-integer
    U_r1 = _round_matrix_to_half_int(best_U)
    V_r1 = _round_matrix_to_half_int(best_V)
    W_r1 = _round_matrix_to_half_int(best_W)
    candidates.append((U_r1, V_r1, W_r1))
    
    # Strategy 2: Round to nearest integer
    U_r2 = np.round(best_U)
    V_r2 = np.round(best_V)
    W_r2 = np.round(best_W)
    candidates.append((U_r2, V_r2, W_r2))
    
    # Strategy 3: Try different offsets for half-integer rounding
    for offset in [0, 0.25, -0.25, 0.125, -0.125]:
        U_r3 = _round_matrix_to_half_int(best_U + offset)
        V_r3 = _round_matrix_to_half_int(best_V + offset)
        W_r3 = _round_matrix_to_half_int(best_W + offset)
        candidates.append((U_r3, V_r3, W_r3))
    
    # Verify each candidate
    for Uc, Vc, Wc in candidates:
        if _verify_fast(Uc.tolist(), Vc.tolist(), Wc.tolist(), n, m, p, atol=1e-9):
            return Uc.tolist(), Vc.tolist(), Wc.tolist()
    
    # Final polish: use rounded values as starting point
    U_f = U_r1.copy()
    V_f = V_r1.copy()
    W_f = W_r1.copy()
    
    for it in range(100):
        if deadline is not None and time.time() > deadline:
            break
        
        _als_iteration()
        
        U_close = np.allclose(U, np.round(U*2)/2.0, atol=1e-6)
        V_close = np.allclose(V, np.round(V*2)/2.0, atol=1e-6)
        W_close = np.allclose(W, np.round(W*2)/2.0, atol=1e-6)
        
        if U_close and V_close and W_close:
            U_r = _round_matrix_to_half_int(U)
            V_r = _round_matrix_to_half_int(V)
            W_r = _round_matrix_to_half_int(W)
            if _verify_fast(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, atol=1e-9):
                return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    return None

def _try_reduce_from_baseline(U_base, V_base, W_base, n, m, p, deadline):
    """Try to reduce the rank starting from a baseline decomposition."""
    Ua = np.array(U_base, dtype=np.float64)
    Va = np.array(V_base, dtype=np.float64)
    Wa = np.array(W_base, dtype=np.float64)
    rank = Ua.shape[1]
    
    best_result = None
    best_rank = rank
    
    for target_rank in range(rank - 1, max(45, rank - 20), -1):
        if time.time() > deadline:
            break
        
        for init_seed in range(6):
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
                np.random.seed(999 + target_rank)
                idx = np.random.choice(rank, target_rank, replace=False)
                U_init = Ua[:, idx] + np.random.randn(n*m, target_rank) * 0.1
                V_init = Va[:, idx] + np.random.randn(m*p, target_rank) * 0.1
                W_init = Wa[:, idx] + np.random.randn(p*n, target_rank) * 0.1
            elif init_seed == 4:
                np.random.seed(2000 + target_rank)
                U_init = Ua[:, :target_rank] + np.random.randn(n*m, target_rank) * 0.05
                V_init = Va[:, :target_rank] + np.random.randn(m*p, target_rank) * 0.05
                W_init = Wa[:, :target_rank] + np.random.randn(p*n, target_rank) * 0.05
            else:
                np.random.seed(3000 + target_rank)
                idx = np.random.choice(rank, target_rank, replace=False)
                U_init = Ua[:, idx] + np.random.randn(n*m, target_rank) * 0.2
                V_init = Va[:, idx] + np.random.randn(m*p, target_rank) * 0.2
                W_init = Wa[:, idx] + np.random.randn(p*n, target_rank) * 0.2
            
            result = _als_decompose(n, m, p, target_rank, 
                                    U_init=U_init, V_init=V_init, W_init=W_init,
                                    max_iters=250, deadline=deadline, polish_iters=150)
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
    deadline = start_time + 140  # Safety margin: evaluator kills at 150s
    
    n, m, p = DIMENSIONS
    
    # Build the Kronecker-embedded Strassen baseline (rank 65)
    U_base, V_base, W_base = _kron_strassen_construct(n, m, p)
    
    # Verify baseline
    if not _verify_fast(U_base, V_base, W_base, n, m, p, atol=1e-9):
        return _schoolbook()
    
    best_result = {"U": U_base, "V": V_base, "W": W_base}
    best_rank = len(U_base[0])
    
    # Try to reduce rank from the baseline
    if time.time() < deadline - 40:
        reduced = _try_reduce_from_baseline(U_base, V_base, W_base, n, m, p, deadline - 40)
        if reduced is not None:
            U_red, V_red, W_red = reduced
            best_result = {"U": U_red, "V": V_red, "W": W_red}
            best_rank = len(U_red[0])
    
    # Try random initializations at various ranks
    for rank in range(best_rank - 1, max(50, best_rank - 15), -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        for seed in range(15):
            if time.time() > deadline - 20:
                break
            
            result = _als_decompose(n, m, p, rank, seed=seed, max_iters=400, 
                                    deadline=deadline - 20, polish_iters=100)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                found = True
                break
        
        if not found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
