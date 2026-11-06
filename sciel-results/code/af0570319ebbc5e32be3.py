import numpy as np
import time
import scipy.optimize as so
import scipy.integrate as si

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit

    Omega_m = 0.3
    Omega_L = 0.7

    # --- 1. Define Training Cosmologies ---
    # Use a diverse but manageable set of training cases to avoid overfitting.
    # Cover the range of w0 and wa likely in validation, and a spread of z.
    w0_train = [-1.5, -1.0, -0.5, 0.0, 0.5]
    wa_train = [-1.5, -0.5, 0.0, 0.5, 1.5]
    z_train = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    train_cases = []
    for w0 in w0_train:
        for wa in wa_train:
            for z in z_train:
                train_cases.append((z, w0, wa))
                
    train_cases = np.array(train_cases)
    N_train = len(train_cases)
    
    # --- 2. Compute Exact Integrals ---
    # Use high-precision adaptive quadrature for the training set.
    # This is done once, so the cost is amortized over the optimization.
    exact_vals = np.zeros(N_train)
    
    def _compute_exact(z, w0, wa):
        def integrand(t):
            a = 1.0 / (1.0 + z * t)
            if a <= 0 or a >= 1:
                return 0.0
            # CPL: Omega_L(a) = Omega_L0 * a^(-3(1+w0+wa)) * (1-a)^(3wa)
            term_m = Omega_m * (a ** -3)
            
            # Log-space for stability
            log_term_L = -3.0 * (1.0 + w0 + wa) * np.log(a) + 3.0 * wa * np.log(1.0 - a)
            log_term_L = np.clip(log_term_L, -700, 700)
            term_L = Omega_L * np.exp(log_term_L)
            
            E2 = term_m + term_L
            if E2 <= 0:
                return 0.0
            return 1.0 / np.sqrt(E2)
        
        try:
            res, _ = si.quad(integrand, 0.0, 1.0, limit=200, epsabs=1e-12, epsrel=1e-12)
            return res
        except Exception:
            return 1.0

    for i in range(N_train):
        if time.time() > deadline - 30:
            break
        z, w0, wa = train_cases[i]
        exact_vals[i] = _compute_exact(z, w0, wa)

    # Filter valid cases
    valid_mask = exact_vals > 1e-6
    if np.sum(valid_mask) < 10:
        # Fallback to Gauss-Legendre if computation fails or is too slow
        nodes, weights = np.polynomial.legendre.leggauss(8)
        nodes = (nodes + 1.0) / 2.0
        weights = weights / 2.0
        return {"nodes": nodes, "weights": weights}
        
    train_cases = train_cases[valid_mask]
    exact_vals = exact_vals[valid_mask]
    N_train = len(train_cases)
    
    z_arr = train_cases[:, 0]
    w0_arr = train_cases[:, 1]
    wa_arr = train_cases[:, 2]

    # --- 3. Optimization Setup ---
    
    def _quadrature_vec(nodes, weights):
        """
        Vectorized quadrature evaluation.
        nodes: (8,)
        weights: (8,)
        returns: (N_train,) quadrature results
        """
        # A: (N_train, 8)
        A = 1.0 / (1.0 + np.outer(z_arr, nodes))
        A = np.clip(A, 1e-12, 1.0 - 1e-12)
        
        # Compute E(a)
        term_m = Omega_m * (A ** -3)
        
        log_term_L = -3.0 * (1.0 + w0_arr[:, None] + wa_arr[:, None]) * np.log(A) + 3.0 * wa_arr[:, None] * np.log(1.0 - A)
        log_term_L = np.clip(log_term_L, -700, 700)
        term_L = Omega_L * np.exp(log_term_L)
        
        E2 = term_m + term_L
        E2 = np.maximum(E2, 1e-300)
        E = np.sqrt(E2)
        inv_E = 1.0 / E
        
        # Quadrature sum: sum_i w_i * inv_E[:, i]
        quad_vals = inv_E @ weights
        return quad_vals

    def _objective(params):
        """
        Minimize the maximum relative error over the training set.
        params: 
        - 0: x0 (logit of node 0)
        - 1-7: log(gaps) between nodes
        - 8-15: log(weights)
        """
        # Unpack nodes
        x0 = params[0]
        log_gaps = params[1:8]
        
        # Ensure gaps are positive and not too small/large
        gaps = np.exp(log_gaps)
        gaps = np.clip(gaps, 1e-6, 10.0) # Reasonable bounds for gaps in logit space? 
        # Actually, nodes are in [0,1], logit space is (-inf, inf).
        # If we define nodes via cumulative sum in logit space:
        # n_0 = sigmoid(x0)
        # n_i = sigmoid(x0 + sum_{j=0}^{i-1} exp(log_gap_j))
        
        x = np.zeros(8)
        x[0] = x0
        for i in range(1, 8):
            x[i] = x[i-1] + gaps[i-1]
            
        nodes = 1.0 / (1.0 + np.exp(-x))
        nodes = np.clip(nodes, 1e-12, 1.0 - 1e-12)
        
        # Check strictly increasing
        if np.any(np.diff(nodes) <= 0):
            return 1e6

        # Unpack weights
        log_w = params[8:16]
        w = np.exp(log_w)
        w_sum = np.sum(w)
        if w_sum < 1e-12:
            return 1e6
        w = w / w_sum
        
        # Evaluate quadrature
        quad_vals = _quadrature_vec(nodes, w)
        
        # Relative error
        rel_errs = np.abs(quad_vals - exact_vals) / exact_vals
        max_err = np.max(rel_errs)
        
        return max_err

    # Initial guess: Gauss-Legendre
    nodes_init, weights_init = np.polynomial.legendre.leggauss(8)
    nodes_init = (nodes_init + 1.0) / 2.0
    weights_init = weights_init / 2.0
    
    # Convert to parameter space
    x_init = np.log(nodes_init / (1.0 - nodes_init))
    params_init = np.zeros(16)
    params_init[0] = x_init[0]
    for i in range(1, 8):
        gap = x_init[i] - x_init[i-1]
        if gap <= 0:
            gap = 1e-3
        params_init[1+i-1] = np.log(gap)
    params_init[8:16] = np.log(weights_init)
    
    best_params = params_init
    best_err = _objective(params_init)
    
    # Define bounds for Differential Evolution
    # x0: logit(0.01) to logit(0.99) approx -4.6 to 4.6
    # gaps: log(1e-4) to log(10) approx -9 to 2.3
    # log_w: log(1e-4) to log(1) approx -9 to 0
    bounds = [(-5.0, 5.0)] + [(-9.0, 3.0)] * 7 + [(-9.0, 1.0)] * 8
    
    # Run Differential Evolution
    # DE is global and robust.
    if time.time() < deadline - 10:
        try:
            # popsize=15 is default, maxiter=1000. 
            # We need to be careful with time.
            # tol=1e-6 might be too strict for time, let's use 1e-5.
            res = so.differential_evolution(
                _objective, 
                bounds, 
                seed=42,
                maxiter=200, 
                tol=1e-5,
                popsize=15,
                mutation=(0.3, 1.0),
                recombination=0.9,
                polish=False  # Polish uses L-BFGS, might be slow or unnecessary if DE is good
            )
            if res.fun < best_err:
                best_err = res.fun
                best_params = res.x
        except Exception as e:
            print(f"DE failed: {e}")

    # Optional: Local polish with Nelder-Mead if time remains
    if time.time() < deadline - 5:
        try:
            res2 = so.minimize(_objective, best_params, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res2.fun < best_err:
                best_err = res2.fun
                best_params = res2.x
        except Exception:
            pass

    # Extract final nodes and weights
    x0 = best_params[0]
    log_gaps = best_params[1:8]
    log_w = best_params[8:16]
    
    gaps = np.exp(log_gaps)
    x = np.zeros(8)
    x[0] = x0
    for i in range(1, 8):
        x[i] = x[i-1] + gaps[i-1]
        
    final_nodes = 1.0 / (1.0 + np.exp(-x))
    final_nodes = np.clip(final_nodes, 1e-12, 1.0 - 1e-12)
    
    final_weights = np.exp(log_w)
    final_weights = final_weights / np.sum(final_weights)
    
    # Final safety checks
    if not np.all(np.diff(final_nodes) > 0):
        # Fallback if numerical issues occurred
        nodes, weights = np.polynomial.legendre.leggauss(8)
        nodes = (nodes + 1.0) / 2.0
        weights = weights / 2.0
        return {"nodes": nodes, "weights": weights}
        
    if not np.all(final_weights > 0):
        nodes, weights = np.polynomial.legendre.leggauss(8)
        nodes = (nodes + 1.0) / 2.0
        weights = weights / 2.0
        return {"nodes": nodes, "weights": weights}

    return {"nodes": final_nodes, "weights": final_weights}
