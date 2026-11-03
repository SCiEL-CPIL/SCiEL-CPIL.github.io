import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def _verify_fast(U, V, W, n=4, m=4, p=5, atol=1e-9):
    """Fast verification using numpy einsum."""
    Ua = np.array(U, dtype=np.float64).reshape(n, m, -1)
    Va = np.array(V, dtype=np.float64).reshape(m, p, -1)
    Wa = np.array(W, dtype=np.float64).reshape(p, n, -1)
    # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    T = np.einsum('ijr,jkr,kir->ijk', Ua, Va, Wa)
    return np.allclose(T, 1.0, atol=atol)

def _round_to_half_integer(x):
    return round(x * 2) / 2.0

def _round_matrix_to_half_int(M):
    return np.array([[_round_to_half_integer(x) for x in row] for row in M])

def _strassen_2x2x2_terms():
    """Return the 7 Strassen 2x2x2 terms as sparse (index, value) lists for U, V, W.
    U indices: (i,j) -> i*2+j
    V indices: (j,k) -> j*2+k
    W indices: (k,i) -> k*2+i
    """
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

def _als_decompose(n, m, p, rank, seed=42, max_iters=200, deadline=None,
                   U_init=None, V_init=None, W_init=None):
    """Fast ALS with vectorized error calculation and batched lstsq."""
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
    
    # Pre-allocate for reshaped views
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U
        # A_{i,j,k,r} = V_{j,k,r} * W_{k,i,r}
        # We want to solve for U_{i,j,r} such that sum_r U_{i,j,r} A_{i,j,k,r} = 1 for all k
        # A is (n, m, p, rank)
        A_U = np.einsum('jkr,kir->ijkr', Vr, Wr)
        b_U = np.ones((n, m, p))
        # Solve batched least squares
        # A_U is (n*m, p, rank). We need to solve A_U[i,j] @ U[i,j] = b_U[i,j]
        # Reshape to (n*m, p, rank) and (n*m, p)
        A_U_flat = A_U.reshape(n*m, p, rank)
        b_U_flat = b_U.reshape(n*m, p)
        U_flat = U.reshape(n*m, rank)
        
        for idx in range(n*m):
            try:
                U_flat[idx] = np.linalg.lstsq(A_U_flat[idx], b_U_flat[idx], rcond=None)[0]
            except:
                pass
        U = U_flat.reshape(n, m, rank).copy().reshape(n*m, rank) # Copy to ensure contiguous? No, just update in place if possible
        # Actually, we updated U_flat which is a view. But we need to update U.
        # Let's just assign back.
        U[:] = U_flat
        
        # Update V
        # A_{j,k,i,r} = U_{i,j,r} * W_{k,i,r}
        # Solve for V_{j,k,r} such that sum_r V_{j,k,r} A_{j,k,i,r} = 1 for all i
        A_V = np.einsum('ijr,kir->jkir', Ur, Wr)
        b_V = np.ones((m, p, n))
        A_V_flat = A_V.reshape(m*p, n, rank)
        b_V_flat = b_V.reshape(m*p, n)
        V_flat = V.reshape(m*p, rank)
        
        for idx in range(m*p):
            try:
                V_flat[idx] = np.linalg.lstsq(A_V_flat[idx], b_V_flat[idx], rcond=None)[0]
            except:
                pass
        V[:] = V_flat
        
        # Update W
        # A_{k,i,j,r} = U_{i,j,r} * V_{j,k,r}
        # Solve for W_{k,i,r} such that sum_r W_{k,i,r} A_{k,i,j,r} = 1 for all j
        A_W = np.einsum('ijr,jkr->kijr', Ur, Vr)
        b_W = np.ones((p, n, m))
        A_W_flat = A_W.reshape(p*n, m, rank)
        b_W_flat = b_W.reshape(p*n, m)
        W_flat = W.reshape(p*n, rank)
        
        for idx in range(p*n):
            try:
                W_flat[idx] = np.linalg.lstsq(A_W_flat[idx], b_W_flat[idx], rcond=None)[0]
            except:
                pass
        W[:] = W_flat
        
        # Calculate Error
        T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
        err = np.linalg.norm(T - 1.0)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if err < 1e-12:
            break
            
    # Round to half-integers
    U_r = _round_matrix_to_half_int(best_U)
    V_r = _round_matrix_to_half_int(best_V)
    W_r = _round_matrix_to_half_int(best_W)
    
    if _verify_fast(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, atol=1e-9):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Build baseline
    U_base, V_base, W_base = _kron_strassen_construct(n, m, p)
    
    if not _verify_fast(U_base, V_base, W_base, n, m, p, atol=1e-9):
        # Fallback to schoolbook if baseline fails (should not happen)
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
        
    best_result = {"U": U_base, "V": V_base, "W": W_base}
    best_rank = len(U_base[0])
    
    # Try to reduce rank
    # We know 65 is possible. Try 64, 63, ...
    for target_rank in range(best_rank - 1, max(50, best_rank - 15), -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Try random initializations
        for seed in range(20):
            if time.time() > deadline - 10:
                break
            result = _als_decompose(n, m, p, target_rank, seed=seed, max_iters=300, deadline=deadline - 10)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                found = True
                break
                
        # Try baseline-derived initialization (drop columns)
        if not found:
            Ua = np.array(U_base, dtype=np.float64)
            Va = np.array(V_base, dtype=np.float64)
            Wa = np.array(W_base, dtype=np.float64)
            # Drop the last (target_rank - best_rank) columns? No, just take first target_rank
            # Or random subset
            for init_seed in range(5):
                if time.time() > deadline - 10:
                    break
                if init_seed == 0:
                    U_init = Ua[:, :target_rank]
                    V_init = Va[:, :target_rank]
                    W_init = Wa[:, :target_rank]
                else:
                    np.random.seed(init_seed * 100 + target_rank)
                    idx = np.random.choice(Ua.shape[1], target_rank, replace=False)
                    U_init = Ua[:, idx]
                    V_init = Va[:, idx]
                    W_init = Wa[:, idx]
                
                result = _als_decompose(n, m, p, target_rank, seed=init_seed, max_iters=300, 
                                        U_init=U_init, V_init=V_init, W_init=W_init, deadline=deadline - 10)
                if result is not None:
                    U, V, W = result
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    found = True
                    break
                    
        if found:
            continue
            
    return best_result
