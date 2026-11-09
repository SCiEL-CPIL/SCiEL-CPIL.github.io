<<<<<<< SEARCH
def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    
    Uses multi-restart L-BFGS-B optimization with fast gradient computation
    plus coordinate descent polish to navigate the nonsmooth landscape.
    """
    start_time = time.time()
    deadline = start_time + 630  # 630 second wall-clock limit
    
    n = 64
    M = 20000  # Full quadrature resolution
    
    # Generate diverse initial points
    x_keich = _keich_initial(n)
    
    # Create multiple initializations
    inits = [x_keich.copy()]
    
    # Add perturbed versions of Keich
    rng = np.random.default_rng(42)
    for trial in range(10):
        scale = 0.01 * (trial + 1)
        x_pert = x_keich + rng.normal(0, scale, size=n)
        inits.append(x_pert)
    
    # Add a few random initializations
    for trial in range(5):
        x_rand = rng.uniform(-0.5, 0.5, size=n)
        inits.append(x_rand)
    
    # Evaluate each initialization and run optimization
    best_x = x_keich.copy()
    best_area = _compute_area_fast(x_keich, n, M)
    
    for init_idx, x0 in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > deadline - 90:  # Leave 90s margin for polish
            break
        
        # Run L-BFGS-B with consistent fun/jac via single compute
        try:
            def fun_and_jac(x_eval):
                area, grad = _compute_area_and_grad(x_eval, n, M)
                return area, grad
            
            result = minimize(
                fun=fun_and_jac,
                x0=x0,
                jac=True,  # fun returns (f, g)
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < best_area:
                area_full = _compute_area_fast(result.x, n, M)
                if area_full < best_area:
                    best_area = area_full
                    best_x = result.x.copy()
        except Exception:
            continue
    
    # Final L-BFGS-B polish with tighter tolerance
    elapsed = time.time() - start_time
    if elapsed < deadline - 60:
        try:
            def fun_and_jac2(x_eval):
                area, grad = _compute_area_and_grad(x_eval, n, M)
                return area, grad
            
            result = minimize(
                fun=fun_and_jac2,
                x0=best_x,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result.fun < best_area:
                area_full = _compute_area_fast(result.x, n, M)
                if area_full < best_area:
                    best_area = area_full
                    best_x = result.x.copy()
        except Exception:
            pass
    
    # Coordinate descent polish to navigate nonsmooth landscape
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        try:
            polished_x, polished_area = _coordinate_descent_polish(
                best_x, n, M, max_rounds=80, deadline=deadline
            )
            if polished_area < best_area:
                best_area = polished_area
                best_x = polished_x.copy()
        except Exception:
            pass
    
    # Ensure valid output
    if not np.all(np.isfinite(best_x)):
        best_x = x_keich
    
    return best_x.astype(np.float64)
=======
def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    
    Uses multi-restart L-BFGS-B optimization with soft-min surrogate gradients
    and increasing beta for convergence to the nonsmooth objective.
    """
    start_time = time.time()
    deadline = start_time + 630  # 630 second wall-clock limit
    
    n = 64
    M = 20000  # Full quadrature resolution
    
    # Generate diverse initial points
    x_keich = _keich_initial(n)
    
    # Create multiple initializations
    inits = [x_keich.copy()]
    
    # Add perturbed versions of Keich
    rng = np.random.default_rng(42)
    for trial in range(10):
        scale = 0.01 * (trial + 1)
        x_pert = x_keich + rng.normal(0, scale, size=n)
        inits.append(x_pert)
    
    # Add a few random initializations
    for trial in range(5):
        x_rand = rng.uniform(-0.5, 0.5, size=n)
        inits.append(x_rand)
    
    # Evaluate each initialization and run optimization
    best_x = x_keich.copy()
    best_area = _compute_area_fast(x_keich, n, M)
    
    # Beta schedule: start smooth, increase toward exact
    beta_schedule = [50.0, 100.0, 200.0, 500.0]
    
    for init_idx, x0 in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > deadline - 90:  # Leave 90s margin for polish
            break
        
        # Run L-BFGS-B with increasing beta
        current_x = x0.copy()
        for beta in beta_schedule:
            elapsed = time.time() - start_time
            if elapsed > deadline - 90:
                break
            
            try:
                def fun_and_jac(x_eval, beta_val=beta):
                    area, grad = _compute_area_and_grad(x_eval, n, M, beta=beta_val)
                    return area, grad
                
                result = minimize(
                    fun=fun_and_jac,
                    x0=current_x,
                    jac=True,  # fun returns (f, g)
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if result.success and result.fun < _compute_area_fast(current_x, n, M):
                    current_x = result.x.copy()
            except Exception:
                break
        
        # Check if this trajectory improved
        area_full = _compute_area_fast(current_x, n, M)
        if area_full < best_area:
            best_area = area_full
            best_x = current_x.copy()
    
    # Final polish with high beta and tight tolerance
    elapsed = time.time() - start_time
    if elapsed < deadline - 60:
        for beta in [500.0, 1000.0]:
            elapsed = time.time() - start_time
            if elapsed > deadline - 30:
                break
            
            try:
                def fun_and_jac2(x_eval, beta_val=beta):
                    area, grad = _compute_area_and_grad(x_eval, n, M, beta=beta_val)
                    return area, grad
                
                result = minimize(
                    fun=fun_and_jac2,
                    x0=best_x,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if result.success:
                    area_full = _compute_area_fast(result.x, n, M)
                    if area_full < best_area:
                        best_area = area_full
                        best_x = result.x.copy()
            except Exception:
                pass
    
    # Coordinate descent polish to navigate nonsmooth landscape
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        try:
            polished_x, polished_area = _coordinate_descent_polish(
                best_x, n, M, max_rounds=80, deadline=deadline
            )
            if polished_area < best_area:
                best_area = polished_area
                best_x = polished_x.copy()
        except Exception:
            pass
    
    # Ensure valid output
    if not np.all(np.isfinite(best_x)):
        best_x = x_keich
    
    return best_x.astype(np.float64)
>>>>>>> REPLACE
