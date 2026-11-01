# ... (existing code) ...

def _try_rank10_polynomial_search(deadline):
    """
    Attempt to find a rank-10 decomposition by treating the bilinear map constraints
    as polynomial equations and using structured numerical solvers with high precision.
    Uses the rank-11 solution as a guide to identify which term to eliminate.
    """
    if time.time() > deadline - 25:
        return None
    
    try:
        from scipy.optimize import least_squares, minimize
    except ImportError:
        return None
    
    base = _rank11_decomposition()
    U11 = np.array(base["U"], dtype=float)
    V11 = np.array(base["V"], dtype=float)
    W11 = np.array(base["W"], dtype=float)
    
    rng = np.random.default_rng(12345)
    best_cost = float('inf')
    best_params = None
    
    # Strategy: Try dropping each term, especially the "simple" ones (7-10)
    # and solve for the rest.
    for drop_term in range(11):
        if time.time() > deadline - 20:
            break
        
        keep = [r for r in range(11) if r != drop_term]
        U0 = U11[:, keep].copy()
        V0 = V11[:, keep].copy()
        W0 = W11[:, keep].copy()
        
        # Multiple perturbations
        for attempt in range(5):
            if time.time() > deadline - 20:
                break
            
            # Scale perturbation by attempt number
            scale = 0.05 * (attempt + 1)
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            x0 = x0 + rng.normal(0, scale, 160)
            
            try:
                # Use least squares first
                result = least_squares(_residuals_rank10, x0, method='lm', max_nfev=20000, ftol=1e-15, xtol=1e-15)
                if result.cost < best_cost:
                    best_cost = result.cost
                    best_params = result.x.copy()
                    if result.cost < 1e-12:
                        break
                        
                # If close, try to polish with Nelder-Mead on the cost function
                if result.cost < 1e-6:
                    def cost_func(x):
                        res = _residuals_rank10(x)
                        return np.sum(res**2)
                    
                    # Nelder-Mead is slower but can find better local minima
                    res2 = minimize(cost_func, result.x, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-15})
                    if res2.fun < best_cost:
                        best_cost = res2.fun
                        best_params = res2.x.copy()
                        if best_cost < 1e-12:
                            break
            except Exception:
                pass
        
        if best_cost < 1e-12:
            break
    
    # Also try random initializations with structured patterns
    for trial in range(30):
        if time.time() > deadline - 20:
            break
        
        # Try small integer/half-integer initializations
        vals = [-1, -0.5, 0, 0.5, 1]
        x0 = np.random.choice(vals, 160).astype(float)
        
        try:
            result = least_squares(_residuals_rank10, x0, method='lm', max_nfev=20000, ftol=1e-15, xtol=1e-15)
            if result.cost < best_cost:
                best_cost = result.cost
                best_params = result.x.copy()
                if best_cost < 1e-12:
                    break
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
    
    # Try different snap strategies
    for snap in [
        lambda x: round(x * 2) / 2,      # half-integers
        lambda x: round(x * 4) / 4,      # quarter-integers
        lambda x: round(x * 8) / 8,      # eighth-integers
        lambda x: round(x),               # integers
        lambda x: round(x * 6) / 6,      # sixths
        lambda x: round(x * 12) / 12,    # twelfths
    ]:
        U = [[snap(u) for u in row] for row in U_mat.tolist()]
        V = [[snap(v) for v in row] for row in V_mat.tolist()]
        W = [[snap(w) for w in row] for row in W_mat.tolist()]
        
        if _verify(U, V, W, 2, 2, 3):
            return {"U": U, "V": V, "W": W}
    
    # If no snap worked, return None (do not return invalid numerical solution)
    return None

def construct():
    # The minimum rank for 2x2x3 matrix multiplication is 11.
    # R(2,2,2) = 7 (Strassen) and R(2,2,1) = 4 (Schoolbook for the third column).
    # Since the B-matrix indices for the two parts are disjoint, the ranks add.
    # Thus, rank 11 is optimal. We return the known rank-11 decomposition.
    
    result11 = _rank11_decomposition()
    
    # Verify to ensure correctness
    if _verify_exact_integer(result11["U"], result11["V"], result11["W"], n, m, p) or \
       _verify(result11["U"], result11["V"], result11["W"], n, m, p):
        
        # Try to find a better (rank 10) decomposition
        start_time = time.time()
        deadline = start_time + 100 # Leave 20s for safety
        
        result10 = _try_rank10_polynomial_search(deadline)
        if result10:
            return result10
            
        return result11
    else:
        return _schoolbook()
