import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def _tensor_values(U, V, W, n, m, p):
    """Vectorized: return the (n*m*p,) array of T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    term = np.einsum('ijr,jkr,kir->ijkr', U3, V3, W3, optimize=True)
    return term.sum(axis=-1).reshape(-1)

def _verify(U, V, W, n, m, p, tol=1e-7):
    vals = _tensor_values(U, V, W, n, m, p)
    return bool(np.all(np.abs(vals - 1.0) < tol))

def _snap_half(x):
    return (2.0 * np.round(2.0 * x)) / 2.0

def _is_half(x, tol=1e-6):
    return bool(np.all(np.abs(2.0 * np.asarray(x, dtype=float) - np.round(2.0 * np.asarray(x, dtype=float))) < tol))

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W

def _block_partition_rank18(n, m, p):
    """
    Construct a rank-18 decomposition by partitioning the 3x6 right matrix into three 3x2 blocks.
    For each block b (b=0,1,2), use the schoolbook rank-6 decomposition for 2x3*3x2.
    """
    n, m, p = 2, 3, 6
    block_p = 2  # each block is 3x2
    num_blocks = p // block_p  # 3 blocks
    
    rank = 6 * num_blocks  # 18
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for b in range(num_blocks):
        offset = b * 6
        # For block b, the local k index k_local in [0, block_p) maps to global k = b*block_p + k_local
        # Local schoolbook for 2x3*3*2:
        for i in range(n):
            for j in range(m):
                for k_local in range(block_p):
                    c = offset + i * m + j * block_p + k_local
                    k_global = b * block_p + k_local
                    U[i * m + j, c] = 1
                    V[j * p + k_global, c] = 1
                    W[k_global * n + i, c] = 1
    
    return U, V, W

def _search_rank(r, n, m, p, budget, seed):
    """Try to find a rank-r decomposition. Returns (U,V,W) list-form or None."""
    nm, mp, pn = n * m, m * p, p * n
    rng = np.random.default_rng(seed)
    target = np.ones(n * m * p)
    
    best_err = np.inf
    best = None
    
    n_restarts = 14
    for rs in range(n_restarts):
        if time.time() > budget:
            break
        # random init
        U = rng.uniform(-1.5, 1.5, (nm, r))
        V = rng.uniform(-1.5, 1.5, (mp, r))
        W = rng.uniform(-1.5, 1.5, (pn, r))
        err = float(np.sum((_tensor_values(U, V, W, n, m, p) - target) ** 2))
        best_local = (err, U.copy(), V.copy(), W.copy())
        
        def solve_W(U, V):
            a = np.einsum('ijr,jkr->ikr', U.reshape(n, m, r), V.reshape(m, p, r), optimize=True)
            Wsol = np.zeros((pn, r))
            for i in range(n):
                for k in range(p):
                    Wsol[k * n + i, :] = np.linalg.lstsq(a[i, k, :].reshape(1, r),
                                                         np.array([1.0]), rcond=None)[0].ravel()
            return Wsol
        
        def solve_U(U, V, W):
            b = np.einsum('jkr,kir->ijr', V.reshape(m, p, r), W.reshape(p, n, r), optimize=True)
            Usol = np.zeros((nm, r))
            for i in range(n):
                for j in range(m):
                    Usol[i * m + j, :] = np.linalg.lstsq(b[i, j, :].reshape(1, r),
                                                         np.array([1.0]), rcond=None)[0].ravel()
            return Usol
        
        def solve_V(U, V, W):
            c = np.einsum('ijr,kir->jkr', U.reshape(n, m, r), W.reshape(p, n, r), optimize=True)
            Vsol = np.zeros((mp, r))
            for j in range(m):
                for k in range(p):
                    Vsol[j * p + k, :] = np.linalg.lstsq(c[j, k, :].reshape(1, r),
                                                         np.array([1.0]), rcond=None)[0].ravel()
            return Vsol
        
        # Iterate ALS
        for it in range(60):
            if time.time() > budget:
                break
            W = solve_W(U, V)
            U = solve_U(U, V, W)
            V = solve_V(U, V, W)
            err = float(np.sum((_tensor_values(U, V, W, n, m, p) - target) ** 2))
            if err < best_local[0]:
                best_local = (err, U.copy(), V.copy(), W.copy())
            if err < 1e-10:
                break
        
        if best_local[0] < best_err:
            best_err = best_local[0]
            best = (best_local[1], best_local[2], best_local[3])
        if best_err < 1e-10:
            break
    
    if best is None:
        return None
    
    U, V, W = best
    # Polish toward half-integers
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        Us = (1 - alpha) * U + alpha * _snap_half(U)
        Vs = (1 - alpha) * V + alpha * _snap_half(V)
        Ws = (1 - alpha) * W + alpha * _snap_half(W)
        e = float(np.sum((_tensor_values(Us, Vs, Ws, n, m, p) - target) ** 2))
        if e < best_err * 1.001 + 1e-12:
            U, V, W, best_err = Us, Vs, Ws, e
    # Final snap attempt
    Uf, Vf, Wf = _snap_half(U), _snap_half(V), _snap_half(W)
    ef = float(np.sum((_tensor_values(Uf, Vf, Wf, n, m, p) - target) ** 2))
    if ef < 1e-6 and _is_half(Uf) and _is_half(Vf) and _is_half(Wf):
        U, V, W = Uf, Vf, Wf
        if _verify(U, V, W, n, m, p, tol=1e-6):
            return U.tolist(), V.tolist(), W.tolist()
    if best_err < 1e-9 and _verify(U, V, W, n, m, p, tol=1e-7):
        if _is_half(_snap_half(U)) and _verify(_snap_half(U), _snap_half(V), _snap_half(W), n, m, p, tol=1e-6):
            return _snap_half(U).tolist(), _snap_half(V).tolist(), _snap_half(W).tolist()
        return U.tolist(), V.tolist(), W.tolist()
    return None

def construct():
    start = time.time()
    deadline = start + 120.0
    n, m, p = DIMENSIONS
    
    U_fb, V_fb, W_fb = _schoolbook(n, m, p)
    
    # First, try the structured rank-18 block partition construction
    U18, V18, W18 = _block_partition_rank18(n, m, p)
    if _verify(U18, V18, W18, n, m, p, tol=1e-7):
        return {"U": U18.tolist(), "V": V18.tolist(), "W": W18.tolist()}
    
    # Search low ranks first. Lower bound is max(n,m,p)=6; true optimum likely ~10-18.
    for r in range(6, 18):
        if time.time() > deadline - 5:
            break
        budget = min(deadline - 3, time.time() + 25)
        res = _search_rank(r, n, m, p, budget, seed=1000 + r)
        if res is not None:
            U, V, W = res
            if _verify(np.array(U), np.array(V), np.array(W), n, m, p, tol=1e-6):
                return {"U": U, "V": V, "W": W}
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
