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
    
    # Add naive 4x4x1 terms for the 5th column (index 4)
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

def _round_to_half_integer(x):
    """Round x to the nearest half-integer."""
    return round(x * 2) / 2.0

def _als_decompose(n, m, p, rank, seed=42, max_iters=300, deadline=None,
                   U_init=None, V_init=None, W_init=None):
    """Try to find a rank-r decomposition using alternating least squares with rounding.
    
    Returns (U, V, W) as lists if found, else None.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize
    if U_init is not None:
        U = np.array(U_init, dtype=np.float64).copy()
        V = np.array(V_init, dtype=np.float64).copy()
        W = np.array(W_init, dtype=np.float64).copy()
    else:
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
    
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update W
        for c in range(rank):
            # Compute residual for W[:, :, c]
            # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
            # For fixed c, we want to minimize sum_{i,j,k} (T[i,j,k] - U[i,j,c] V[j,k,c] W[k,i,c])^2
            # This is a linear system in W[:, :, c]
            
            # Residual = T - sum_{c' != c} U[:,:,c'] V[:,:,c'] W[:,:,c']
            # But we can compute the "target" for W[:,:,c] as:
            # For each (k,i), target = sum_{j} T[i,j,k] / (U[i,j,c] * V[j,k,c]) ... no that's not right.
            
            # Let's do it properly:
            # T[i,j,k] = U[i,j,c] V[j,k,c] W[k,i,c] + R[i,j,k] where R is residual from other terms
            # We want to find W[k,i,c] to minimize sum_{i,j,k} (T[i,j,k] - U[i,j,c] V[j,k,c] W[k,i,c])^2
            
            # For each (k,i), the coefficient is sum_j U[i,j,c] V[j,k,c]
            # The target is sum_j T[i,j,k] ... no, we need to account for other terms.
            
            # Let's compute the full residual first
            pass
        
        # Simpler approach: use the standard ALS for tensor decomposition
        # Residual R = T - sum_c U[:,:,c] ⊙ V[:,:,c] ⊙ W[:,:,c] (element-wise product, summed over c)
        # For updating W[:,:,c]:
        # W[:,:,c] should minimize sum_{i,j,k} (R[i,j,k] + U[i,j,c] V[j,k,c] W[k,i,c])^2
        # = sum_{i,k} (sum_j U[i,j,c] V[j,k,c])^2 W[k,i,c]^2 - 2 sum_{i,k} (sum_j U[i,j,c] V[j,k,c]) (sum_j R[i,j,k] ... )
        
        # Actually, let's just do a simpler gradient-based approach or use the fact that
        # for each (k,i), the optimal W[k,i,c] = (sum_j R[i,j,k] * U[i,j,c] * V[j,k,c]) / (sum_j (U[i,j,c] * V[j,k,c])^2)
        
        # Compute residual
        R = np.ones((n, m, p), dtype=np.float64)
        for cc in range(rank):
            uc = Ur[:, :, cc]
            vc = Vr[:, :, cc]
            wc = Wr[:, :, cc]
            for j in range(m):
                R[:, j, :] = R[:, j, :] - np.outer(uc[:, j], vc[j, :]) * wc.T
        
        # Update W
        for c in range(rank):
            uc = Ur[:, :, c]
            vc = Vr[:, :, c]
            for k in range(p):
                for i in range(n):
                    num = 0.0
                    den = 0.0
                    for j in range(m):
                        uval = uc[i, j]
                        vval = vc[j, k]
                        num += R[i, j, k] * uval * vval
                        den += (uval * vval) ** 2
                    if den > 1e-12:
                        Wr[k, i, c] = num / den
        
        # Update U
        R = np.ones((n, m, p), dtype=np.float64)
        for cc in range(rank):
            uc = Ur[:, :, cc]
            vc = Vr[:, :, cc]
            wc = Wr[:, :, cc]
            for j in range(m):
                R[:, j, :] = R[:, j, :] - np.outer(uc[:, j], vc[j, :]) * wc.T
        
        for c in range(rank):
            vc = Vr[:, :, c]
            wc = Wr[:, :, c]
            for i in range(n):
                for j in range(m):
                    num = 0.0
                    den = 0.0
                    for k in range(p):
                        vval = vc[j, k]
                        wval = wc[k, i]
                        num += R[i, j, k] * vval * wval
                        den += (vval * wval) ** 2
                    if den > 1e-12:
                        Ur[i, j, c] = num / den
        
        # Update V
        R = np.ones((n, m, p), dtype=np.float64)
        for cc in range(rank):
            uc = Ur[:, :, cc]
            vc = Vr[:, :, cc]
            wc = Wr[:, :, cc]
            for j in range(m):
                R[:, j, :] = R[:, j, :] - np.outer(uc[:, j], vc[j, :]) * wc.T
        
        for c in range(rank):
            uc = Ur[:, :, c]
            wc = Wr[:, :, c]
            for j in range(m):
                for k in range(p):
                    num = 0.0
                    den = 0.0
                    for i in range(n):
                        uval = uc[i, j]
                        wval = wc[k, i]
                        num += R[i, j, k] * uval * wval
                        den += (uval * wval) ** 2
                    if den > 1e-12:
                        Vr[j, k, c] = num / den
        
        # Round to half-integers and check
        U_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in U])
        V_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in V])
        W_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in W])
        
        if _verify_fast(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p, atol=1e-9):
            return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
    # Final check on unrounded
    if _verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p, atol=1e-9):
        return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _reduce_rank_via_als(n, m, p, U_init, V_init, W_init, target_rank, deadline=None):
    """Try to reduce rank from current to target_rank using ALS."""
    U = np.array(U_init, dtype=np.float64)
    V = np.array(V_init, dtype=np.float64)
    W = np.array(W_init, dtype=np.float64)
    
    current_rank = U.shape[1]
    
    for target in range(target_rank, current_rank + 1):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try to find a rank-`target` decomposition
        result = _als_decompose(n, m, p, target, seed=42, max_iters=100, deadline=deadline,
                                U_init=U[:, :target], V_init=V[:, :target], W_init=W[:, :target])
        if result is not None:
            return result
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Build the rank 65 baseline
    U_kron, V_kron, W_kron = _kron_strassen_construct(n, m, p)
    kron_ok = _verify_fast(U_kron, V_kron, W_kron, n, m, p, atol=1e-9)
    
    if not kron_ok:
        return _schoolbook()
    
    best_result = {"U": U_kron, "V": V_kron, "W": W_kron}
    best_rank = 65
    
    # Try to reduce rank using ALS
    for target_rank in [60, 55, 50, 45]:
        if time.time() > deadline:
            break
        
        result = _reduce_rank_via_als(n, m, p, U_kron, V_kron, W_kron, target_rank, deadline=deadline)
        if result is not None:
            U_new, V_new, W_new = result
            if _verify_fast(U_new, V_new, W_new, n, m, p, atol=1e-9):
                new_rank = np.array(U_new).shape[1]
                if new_rank < best_rank:
                    best_result = {"U": U_new, "V": V_new, "W": W_new}
                    best_rank = new_rank
    
    return best_result

# EVOLVE-BLOCK-END
