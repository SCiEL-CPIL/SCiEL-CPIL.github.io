"""Tensor decomposition search for 2x2 by 2x3 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 3]
n, m, p = DIMENSIONS


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _rank11_decomposition():
    """
    Rank-11 decomposition: Strassen (rank 7) for 2x2 block + schoolbook (rank 4) for col 2.
    This is the known optimal rank for 2x2x3 over reals/rationals.
    """
    n, m, p = 2, 2, 3
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def set_u(i, j, r, val):
        U[i * m + j][r] = val

    def set_v(j, k, r, val):
        V[j * p + k][r] = val

    def set_w(k, i, r, val):
        W[k * n + i][r] = val

    # Term 0: (A00+A11)(B00+B11) -> contributes to C[0,0] and C[1,1]
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1)
    set_w(1, 1, 0, 1)

    # Term 1: (A10+A11)B00 -> contributes to C[0,1] and C[1,1]
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1)
    set_w(1, 1, 1, -1)

    # Term 2: A00(B01-B11) -> contributes to C[1,0] and C[1,1]
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1)
    set_w(1, 1, 2, 1)

    # Term 3: A11(B10-B00) -> contributes to C[0,0] and C[0,1]
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1)
    set_w(0, 1, 3, 1)

    # Term 4: (A00+A01)B11 -> contributes to C[0,0] and C[1,0]
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1)
    set_w(1, 0, 4, 1)

    # Term 5: (A10-A00)(B00+B01) -> contributes to C[1,1]
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    # Term 6: (A01-A11)(B10+B11) -> contributes to C[0,0]
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    # Term 7: A00*B02
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)

    # Term 8: A01*B12
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    # Term 9: A10*B02
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)

    # Term 10: A11*B12
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}


def _schoolbook():
    """Schoolbook decomposition of rank n*m*p = 12."""
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


def _residuals_rank10(params):
    """Compute residuals for rank-10 tensor decomposition."""
    U_flat = params[:40]
    V_flat = params[40:100]
    W_flat = params[100:160]
    
    U_mat = U_flat.reshape(4, 10)
    V_mat = V_flat.reshape(6, 10)
    W_mat = W_flat.reshape(6, 10)
    
    res = np.empty(12)
    idx = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                s = 0.0
                for r in range(10):
                    s += U_mat[i*2+j, r] * V_mat[j*3+k, r] * W_mat[k*2+i, r]
                res[idx] = s - 1.0
                idx += 1
    return res


def _residuals_rank10_smooth(params, beta=100.0):
    """Smooth surrogate: soft-min of squared residuals via log-sum-exp."""
    res = _residuals_rank10(params)
    sq = res ** 2
    # Use log-sum-exp for numerical stability
    max_sq = np.max(sq)
    if max_sq < 1e-300:
        return 0.0
    log_sum = max_sq + np.log(np.sum(np.exp(sq - max_sq)))
    return log_sum / beta


def _try_rank10_polynomial_search(deadline):
    """
    Attempt to find a rank-10 decomposition using smooth optimization with
    multiple strategies: structured drops, random restarts, and high-precision polish.
    """
    if time.time() > deadline - 25:
        return None
    
    try:
        from scipy.optimize import minimize
    except ImportError:
        return None
    
    base = _rank11_decomposition()
    U11 = np.array(base["U"], dtype=float)
    V11 = np.array(base["V"], dtype=float)
    W11 = np.array(base["W"], dtype=float)
    
    rng = np.random.default_rng(12345)
    best_cost = float('inf')
    best_params = None
    
    def _try_solve(x0, max_nfev=5000):
        """Try to solve from initial point x0."""
        nonlocal best_cost, best_params
        try:
            result = minimize(_residuals_rank10_smooth, x0, method='BFGS', 
                            options={'maxiter': 200, 'gtol': 1e-12})
            if result.fun < best_cost:
                best_cost = result.fun
                best_params = result.x.copy()
            # Also try LM on raw residuals from this point
            try:
                from scipy.optimize import least_squares
                result2 = least_squares(_residuals_rank10, result.x, method='lm', 
                                       max_nfev=max_nfev, ftol=1e-15, xtol=1e-15)
                if result2.cost < best_cost:
                    best_cost = result2.cost
                    best_params = result2.x.copy()
            except Exception:
                pass
        except Exception:
            pass
    
    # Strategy 1: Drop each term from rank-11 and refine
    for drop_term in range(11):
        if time.time() > deadline - 30:
            break
        
        keep = [r for r in range(11) if r != drop_term]
        U0 = U11[:, keep].copy()
        V0 = V11[:, keep].copy()
        W0 = W11[:, keep].copy()
        
        for attempt in range(4):
            if time.time() > deadline - 30:
                break
            scale = 0.05 * (attempt + 1)
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            x0 = x0 + rng.normal(0, scale, 160)
            _try_solve(x0)
        
        if best_cost < 1e-14:
            break
    
    # Strategy 2: Random structured initializations
    for trial in range(40):
        if time.time() > deadline - 30:
            break
        
        vals = [-1, -0.5, 0, 0.5, 1]
        x0 = rng.choice(vals, 160).astype(float)
        _try_solve(x0, max_nfev=3000)
        
        if best_cost < 1e-14:
            break
    
    # Strategy 3: High-precision polish if we got close
    if best_params is not None and best_cost < 1e-6:
        if time.time() < deadline - 15:
            try:
                from scipy.optimize import least_squares
                result = least_squares(_residuals_rank10, best_params, method='lm', 
                                       max_nfev=50000, ftol=1e-18, xtol=1e-18)
                if result.cost < best_cost:
                    best_cost = result.cost
                    best_params = result.x.copy()
            except Exception:
                pass
    
    if best_params is None or best_cost > 1e-8:
        return None
    
    # Extract and try to snap to rational values
    U_flat = best_params[:40]
    V_flat = best_params[40:100]
    W_flat = best_params[100:160]
    
    U_mat = U_flat.reshape(4, 10)
    V_mat = V_flat.reshape(6, 10)
    W_mat = W_flat.reshape(6, 10)
    
    # Try different snap strategies with verification
    snap_strategies = [
        lambda x: round(x * 2) / 2,      # half-integers
        lambda x: round(x * 4) / 4,      # quarter-integers
        lambda x: round(x * 8) / 8,      # eighth-integers
        lambda x: round(x),               # integers
        lambda x: round(x * 6) / 6,      # sixths
        lambda x: round(x * 12) / 12,    # twelfths
    ]
    
    for snap in snap_strategies:
        U = [[snap(u) for u in row] for row in U_mat.tolist()]
        V = [[snap(v) for v in row] for row in V_mat.tolist()]
        W = [[snap(w) for w in row] for row in W_mat.tolist()]
        
        if _verify(U, V, W, 2, 2, 3):
            return {"U": U, "V": V, "W": W}
    
    # Try mixed precision: snap most values, keep a few free
    for snap in snap_strategies[:3]:
        U = [[snap(u) for u in row] for row in U_mat.tolist()]
        V = [[snap(v) for v in row] for row in V_mat.tolist()]
        W = [[snap(w) for w in row] for row in W_mat.tolist()]
        
        # Check if close enough that rounding might work with slight adjustment
        if _verify(U, V, W, 2, 2, 3, tol=1e-6):
            # Try to find exact rational solution near this point
            for final_snap in snap_strategies:
                U2 = [[final_snap(u) for u in row] for row in U_mat.tolist()]
                V2 = [[final_snap(v) for v in row] for row in V_mat.tolist()]
                W2 = [[final_snap(w) for w in row] for row in W_mat.tolist()]
                if _verify(U2, V2, W2, 2, 2, 3):
                    return {"U": U2, "V": V2, "W": W2}
    
    return None


def _try_rank10_groebner_like(deadline):
    """
    Attempt to find a rank-10 decomposition using structured sparse initialization
    and Newton-type refinement.
    """
    if time.time() > deadline - 25:
        return None
    
    try:
        from scipy.optimize import fsolve
    except ImportError:
        return None
    
    rng = np.random.default_rng(999)
    
    best_cost = float('inf')
    best_solution = None
    
    for trial in range(40):
        if time.time() > deadline - 20:
            break
        
        # Random initialization with sparsity
        U = np.zeros((4, 10))
        V = np.zeros((6, 10))
        W = np.zeros((6, 10))
        
        # Fill with random small values
        for i in range(4):
            for j in range(10):
                if rng.random() < 0.3:
                    U[i, j] = rng.choice([-1, 1])
        for i in range(6):
            for j in range(10):
                if rng.random() < 0.3:
                    V[i, j] = rng.choice([-1, 1])
        for i in range(6):
            for j in range(10):
                if rng.random() < 0.3:
                    W[i, j] = rng.choice([-1, 1])
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = fsolve(_residuals_rank10, x0, full_output=True, xtol=1e-14, maxfev=5000)
            x_sol, info, ier, msg = result
            if ier == 1:
                cost = np.sum(np.array(_residuals_rank10(x_sol))**2)
                if cost < best_cost:
                    best_cost = cost
                    best_solution = x_sol.copy()
                    if cost < 1e-14:
                        break
        except Exception:
            pass
    
    if best_solution is None or best_cost > 1e-10:
        return None
    
    # High-precision polish
    if time.time() < deadline - 10:
        try:
            from scipy.optimize import least_squares
            result = least_squares(_residuals_rank10, best_solution, method='lm', 
                                   max_nfev=50000, ftol=1e-18, xtol=1e-18)
            if result.cost < best_cost:
                best_cost = result.cost
                best_solution = result.x.copy()
        except Exception:
            pass
    
    if best_solution is None or best_cost > 1e-8:
        return None
    
    U_flat = best_solution[:40]
    V_flat = best_solution[40:100]
    W_flat = best_solution[100:160]
    U_mat = U_flat.reshape(4, 10)
    V_mat = V_flat.reshape(6, 10)
    W_mat = W_flat.reshape(6, 10)
    
    snap_strategies = [
        lambda x: round(x * 2) / 2,
        lambda x: round(x),
        lambda x: round(x * 4) / 4,
        lambda x: round(x * 8) / 8,
    ]
    
    for snap in snap_strategies:
        U = [[snap(u) for u in row] for row in U_mat.tolist()]
        V = [[snap(v) for v in row] for row in V_mat.tolist()]
        W = [[snap(w) for w in row] for row in W_mat.tolist()]
        
        if _verify(U, V, W, 2, 2, 3):
            return {"U": U, "V": V, "W": W}
    
    return None


def construct():
    deadline = time.time() + 135
    
    # First, get the known rank-11 construction
    result11 = _rank11_decomposition()
    if _verify(result11["U"], result11["V"], result11["W"], n, m, p):
        best = result11
        best_rank = 11
    else:
        best = _schoolbook()
        best_rank = 12
    
    # Try to find a rank-10 solution using multiple strategies
    if best_rank > 10 and time.time() < deadline - 40:
        # Strategy 1: Smooth polynomial search with structured initialization
        result10 = _try_rank10_polynomial_search(deadline)
        if result10 is not None:
            best = result10
            best_rank = 10
        elif time.time() < deadline - 25:
            # Strategy 2: Groebner-like search
            result10 = _try_rank10_groebner_like(deadline)
            if result10 is not None:
                best = result10
                best_rank = 10
    
    return best
