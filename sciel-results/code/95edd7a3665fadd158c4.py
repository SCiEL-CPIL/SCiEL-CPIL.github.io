"""Improved program for matrix multiplication tensor decomposition."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START

def _triples(n, m, p):
    """List of (u_index, v_index, w_index) for each (i,j,k)."""
    tr = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                tr.append((u, v, w))
    return tr

def _residual_fast(U, V, W, tr):
    """Vectorized max abs residual."""
    max_err = 0.0
    for (u, v, w) in tr:
        s = np.dot(U[u] * V[v], W[w])
        e = abs(s - 1.0)
        if e > max_err:
            max_err = e
    return max_err

def _round_to_halfint(A, tol=1e-6):
    """Round to nearest half-integer if close enough, else return None."""
    A2 = A * 2.0
    A_rounded = np.round(A2)
    if np.max(np.abs(A2 - A_rounded)) < tol:
        return A_rounded / 2.0
    return None

def _verify_exact(U, V, W, tr, tol=1e-9):
    """Check if decomposition is exact."""
    return _residual_fast(U, V, W, tr) < tol

def _als_step_fast(U, V, W, tr, n, m, p):
    """One ALS iteration with vectorized operations."""
    r = U.shape[1]
    U_new = U.copy()
    V_new = V.copy()
    W_new = W.copy()
    
    # Precompute maps
    u_map = {}
    v_map = {}
    w_map = {}
    for (u, v, w) in tr:
        u_map.setdefault(u, []).append((v, w))
        v_map.setdefault(v, []).append((u, w))
        w_map.setdefault(w, []).append((u, v))
    
    # Solve U
    for u, lst in u_map.items():
        A = np.array([V[v] * W[w] for (v, w) in lst])
        b = np.ones(len(lst))
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        U_new[u] = sol
    
    # Solve V
    for v, lst in v_map.items():
        A = np.array([U_new[u] * W[w] for (u, w) in lst])
        b = np.ones(len(lst))
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        V_new[v] = sol
    
    # Solve W
    for w, lst in w_map.items():
        A = np.array([U_new[u] * V_new[v] for (u, v) in lst])
        b = np.ones(len(lst))
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        W_new[w] = sol
    
    return U_new, V_new, W_new

def _flatten_factors(U, V, W):
    """Flatten U, V, W into a single vector for optimization."""
    return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

def _unflatten_factors(x, U_shape, V_shape, W_shape):
    """Unflatten a single vector into U, V, W."""
    U_size = U_shape[0] * U_shape[1]
    V_size = V_shape[0] * V_shape[1]
    W_size = W_shape[0] * W_shape[1]
    
    U = x[:U_size].reshape(U_shape)
    V = x[U_size:U_size+V_size].reshape(V_shape)
    W = x[U_size+V_size:].reshape(W_shape)
    return U, V, W

def _objective(x, tr, U_shape, V_shape, W_shape):
    """Objective function for L-BFGS: sum of squared residuals."""
    U, V, W = _unflatten_factors(x, U_shape, V_shape, W_shape)
    err_sq = 0.0
    for (u, v, w) in tr:
        s = np.dot(U[u] * V[v], W[w])
        err_sq += (s - 1.0) ** 2
    return err_sq

def _refine_with_lbfgs(U, V, W, tr, U_shape, V_shape, W_shape, max_iter=1000, deadline=None):
    """Refine factors using L-BFGS."""
    x0 = _flatten_factors(U, V, W)
    
    def obj(x):
        if deadline is not None and time.time() > deadline:
            return float('inf')
        return _objective(x, tr, U_shape, V_shape, W_shape)
    
    try:
        res = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': 1e-12})
        x_opt = res.x
        U_opt, V_opt, W_opt = _unflatten_factors(x_opt, U_shape, V_shape, W_shape)
        return U_opt, V_opt, W_opt
    except Exception:
        return U, V, W

def _search_rank(r, n, m, p, seed, deadline):
    """Search for a rank-r decomposition using ALS and L-BFGS."""
    tr = _triples(n, m, p)
    nm, mp, pn = n * m, m * p, p * n
    U_shape = (nm, r)
    V_shape = (mp, r)
    W_shape = (pn, r)
    
    rng = np.random.RandomState(seed)
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Try a few random initializations
    for trial in range(5):
        if time.time() > deadline:
            break
        
        # Initialize with small random values
        U = rng.randn(*U_shape) * 0.5
        V = rng.randn(*V_shape) * 0.5
        W = rng.randn(*W_shape) * 0.5
        
        # Phase 1: ALS
        for _ in range(50):
            if time.time() > deadline:
                break
            U, V, W = _als_step_fast(U, V, W, tr, n, m, p)
            err = _residual_fast(U, V, W, tr)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-8:
                break
        
        # Phase 2: L-BFGS Refinement
        if time.time() < deadline - 5:
            U, V, W = _refine_with_lbfgs(U, V, W, tr, U_shape, V_shape, W_shape, max_iter=200, deadline=deadline)
            err = _residual_fast(U, V, W, tr)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-8:
                break

    return best_err, best_U, best_V, best_W

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    tr = _triples(n, m, p)
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    sU, sV, sW = _schoolbook()
    best_result = (100, sU, sV, sW)  # (rank, U, V, W)
    
    # Try ranks from low to high
    # The rank of 4x5x5 is likely between 40 and 80.
    candidate_ranks = [50, 55, 60, 65, 70, 75, 80, 85, 90]
    seeds = [42, 1, 7, 123, 999, 555, 777, 111, 222, 333]
    
    for r in candidate_ranks:
        if time.time() > deadline - 10:
            break
        if r >= best_result[0]:
            break
            
        for seed in seeds:
            if time.time() > deadline - 5:
                break
            
            err, U, V, W = _search_rank(r, n, m, p, seed, deadline - 5)
            if U is not None and err < 1e-6:
                # Try to round to half-integers
                U_h = _round_to_halfint(U, 1e-3)
                V_h = _round_to_halfint(V, 1e-3)
                W_h = _round_to_halfint(W, 1e-3)
                if U_h is not None and V_h is not None and W_h is not None:
                    if _verify_exact(U_h, V_h, W_h, tr, 1e-6):
                        if r < best_result[0]:
                            best_result = (r, U_h, V_h, W_h)
                            # If we found a very good one, we might stop early
                            if r < 60:
                                return {"U": best_result[1].tolist(), "V": best_result[2].tolist(), "W": best_result[3].tolist()}
    
    # If no better solution found, return schoolbook
    r, U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
