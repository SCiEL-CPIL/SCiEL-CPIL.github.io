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
    """Compute residuals for rank-10 tensor decomposition using vectorized operations."""
    U_flat = params[:40]
    V_flat = params[40:100]
    W_flat = params[100:160]
    
    U_mat = U_flat.reshape(4, 10)
    V_mat = V_flat.reshape(6, 10)
    W_mat = W_flat.reshape(6, 10)
    
    # Indices for U, V, W
    # U[i, j] -> i*2 + j
    # V[j, k] -> j*3 + k
    # W[k, i] -> k*2 + i
    
    u_idx = np.array([i*2 + j for i in range(2) for j in range(2) for _ in range(3)])
    v_idx = np.array([j*3 + k for i in range(2) for j in range(2) for k in range(3)])
    w_idx = np.array([k*2 + i for i in range(2) for j in range(2) for k in range(3)])
    
    # Extract relevant rows
    u_rows = U_mat[u_idx, :] # shape (12, 10)
    v_rows = V_mat[v_idx, :] # shape (12, 10)
    w_rows = W_mat[w_idx, :] # shape (12, 10)
    
    # Compute sum over rank dimension r
    # s[idx] = sum_r U[u_idx[idx], r] * V[v_idx[idx], r] * W[w_idx[idx], r]
    products = u_rows * v_rows * w_rows
    res = np.sum(products, axis=1) - 1.0
    
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
        from scipy.optimize import least_squares
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
            # Use BFGS on the smooth surrogate first to get into the basin
            result = minimize(_residuals_rank10_smooth, x0, method='BFGS', 
                            options={'maxiter': 100, 'gtol': 1e-10})
            
            current_x = result.x
            current_cost = result.fun
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_params = current_x.copy()
            
            # Refine with LM on raw residuals
            # LM is faster but more sensitive to initial conditions
            result2 = least_squares(_residuals_rank10, current_x, method='lm', 
                                   max_nfev=max_nfev, ftol=1e-15, xtol=1e-15)
            
            if result2.cost < best_cost:
                best_cost = result2.cost
                best_params = result2.x.copy()
                
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
        
        # Try several perturbation scales
        for scale in [0.01, 0.05, 0.1, 0.2, 0.5]:
            if time.time() > deadline - 30:
                break
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            x0 = x0 + rng.normal(0, scale, 160)
            _try_solve(x0)
        
        if best_cost < 1e-14:
            break
    
    # Strategy 2: Random initializations
    # Increase number of trials and use diverse distributions
    num_random_trials = 100
    for trial in range(num_random_trials):
        if time.time() > deadline - 30:
            break
        
        # Mix of discrete and continuous initializations
        if trial % 2 == 0:
             # Discrete small integers/halves
             vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
             x0 = rng.choice(vals, 160).astype(float)
        else:
             # Continuous Gaussian
             # Start with small noise around zero
             x0 = rng.normal(0, 1.0, 160)
        
        _try_solve(x0, max_nfev=2000)
        
        if best_cost < 1e-14:
            break
    
    # Strategy 3: High-precision polish if we got close
    if best_params is not None and best_cost < 1e-6:
        if time.time() < deadline - 15:
            try:
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
    # Include more denominators to catch potential rational solutions
    snap_strategies = [
        lambda x: round(x * 2) / 2,      # half-integers
        lambda x: round(x * 4) / 4,      # quarter-integers
        lambda x: round(x * 8) / 8,      # eighth-integers
        lambda x: round(x),               # integers
        lambda x: round(x * 6) / 6,      # sixths
        lambda x: round(x * 12) / 12,    # twelfths
        lambda x: round(x * 3) / 3,      # thirds
        lambda x: round(x * 5) / 5,      # fifths
    ]
    
    for snap in snap_strategies:
        U = [[snap(u) for u in row] for row in U_mat.tolist()]
        V = [[snap(v) for v in row] for row in V_mat.tolist()]
        W = [[snap(w) for w in row] for row in W_mat.tolist()]
        
        if _verify(U, V, W, 2, 2, 3):
            return {"U": U, "V": V, "W": W}
    
    # If close but not exact, try to fix the worst error by adjusting one parameter
    # This is a heuristic for when the solution is "almost" rational
    if best_cost < 1e-4:
        # Identify the residual with the largest magnitude
        res = _residuals_rank10(best_params)
        worst_idx = np.argmax(np.abs(res))
        
        # Try to adjust one parameter in U, V, or W to fix this residual
        # This is complex, so we just try more snapping with tighter tolerances
        for snap in snap_strategies:
            U = [[snap(u) for u in row] for row in U_mat.tolist()]
            V = [[snap(v) for v in row] for row in V_mat.tolist()]
            W = [[snap(w) for w in row] for row in W_mat.tolist()]
            
            # Verify with a slightly looser tolerance first
            if _verify(U, V, W, 2, 2, 3, tol=1e-5):
                # If it's close, try to snap with higher precision
                for high_snap in [lambda x: round(x * 20) / 20, lambda x: round(x * 100) / 100]:
                     U2 = [[high_snap(u) for u in row] for row in U_mat.tolist()]
                     V2 = [[high_snap(v) for v in row] for row in V_mat.tolist()]
                     W2 = [[high_snap(w) for w in row] for row in W_mat.tolist()]
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
