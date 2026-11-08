import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar
from scipy.special import eval_genlaguerre, roots_genlaguerre
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin before 630s kill

    m = 12
    alpha = 1

    # Precompute the transformation matrix from monomial basis to Laguerre basis
    # L_k^(alpha)(x) = sum_{j=0}^k A[k,j] x^j
    # For alpha=1: A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    A = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k + alpha, j + 1) / math.factorial(j)

    # Inverse: convert monomial coefficients to Laguerre coefficients
    A_inv = np.linalg.inv(A)

    def laguerre_norm_sq(c):
        """
        Compute the squared norm of the polynomial given its Laguerre coefficients c.
        The norm is sum_{k=0}^{m} c[k]^2 / (k + alpha)
        """
        return float(np.sum(c**2 / (np.arange(m + 1) + alpha)))

    def poly_from_laguerre(c, x):
        """
        Evaluate polynomial with Laguerre coefficients c at points x.
        P(x) = sum_{k=0}^{m} c[k] * L_k^(alpha)(x)
        """
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        for k in range(m + 1):
            if c[k] != 0:
                result += c[k] * eval_genlaguerre(k, alpha, x)
        return result

    def poly_derivative_from_laguerre(c, x):
        """
        Evaluate derivative of polynomial with Laguerre coefficients c at points x.
        d/dx L_k^(alpha)(x) = -L_{k-1}^{(alpha+1)}(x)
        """
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        for k in range(1, m + 1):
            if c[k] != 0:
                result += c[k] * (-eval_genlaguerre(k - 1, alpha + 1, x))
        return result

    def penalty_for_double_roots(c, roots):
        """
        Compute penalty for not having double roots at the given locations.
        Penalty = sum_i [P(r_i)^2 + P'(r_i)^2]
        """
        p_vals = poly_from_laguerre(c, roots)
        dp_vals = poly_derivative_from_laguerre(c, roots)
        return float(np.sum(p_vals**2) + np.sum(dp_vals**2))

    def cost_func(params):
        """
        params: [c_0, ..., c_12, log_r_0, ..., log_r_11]
        First m+1 elements are Laguerre coefficients.
        Next m elements are log of root locations.
        
        Objective: minimize 1/||P||^2 + lambda * penalty
        """
        c = params[:m + 1]
        log_roots = params[m + 1:]
        roots = np.exp(log_roots)
        
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
            
        norm_sq = laguerre_norm_sq(c)
        if norm_sq <= 1e-10:
            return 1e10
            
        # Penalty for double root constraint
        pen = penalty_for_double_roots(c, roots)
        
        # Combined objective: want small 1/norm_sq AND small penalty
        # Use a weighted combination
        lambda_pen = 1000.0
        return 1.0 / norm_sq + lambda_pen * pen

    # Get initial Laguerre coefficients from L_12^(1)(x) zeros
    try:
        laguerre_roots = roots_genlaguerre(m, alpha)[0]
        laguerre_roots = np.sort(laguerre_roots)
        laguerre_roots = np.maximum(laguerre_roots, 1e-10)
        
        # Build polynomial P(x) = prod(x - r_i) and convert to Laguerre basis
        coeffs_desc = np.poly(laguerre_roots)
        p_asc = coeffs_desc[::-1]
        c_init = A_inv @ p_asc
    except Exception:
        c_init = np.zeros(m + 1)
        c_init[0] = 1.0
        laguerre_roots = np.array([4.0 * (2.0 ** i) for i in range(12)])

    # Initial parameters
    log_roots_init = np.log(laguerre_roots)
    params_init = np.concatenate([c_init, log_roots_init])

    # Define bounds
    # Laguerre coefficients: reasonable range
    c_bounds = [(-100, 100)] * (m + 1)
    # Log roots: [1e-4, 1e4] -> [-9.2, 9.2]
    log_low = -9.0
    log_high = 10.0
    log_bounds = [(log_low, log_high)] * m
    bounds = c_bounds + log_bounds

    best_val = 1e10
    best_params = None
    best_roots = None

    # Run L-BFGS-B from multiple starting points
    rng = np.random.default_rng(42)
    
    initial_guesses = []
    
    # 1. Initial guess from Laguerre zeros
    initial_guesses.append(params_init.copy())
    
    # 2. Perturbed versions
    for scale in [0.5, 0.8, 1.2, 1.5]:
        perturbed_roots = laguerre_roots * scale
        perturbed_roots = np.maximum(perturbed_roots, 1e-8)
        coeffs_desc = np.poly(perturbed_roots)
        p_asc = coeffs_desc[::-1]
        c_pert = A_inv @ p_asc
        log_roots_pert = np.log(perturbed_roots)
        params_pert = np.concatenate([c_pert, log_roots_pert])
        initial_guesses.append(params_pert)
    
    # 3. Random perturbations in Laguerre coefficient space
    for _ in range(20):
        c_rand = c_init + rng.normal(0, 0.1, size=m + 1)
        log_roots_rand = log_roots_init + rng.normal(0, 0.5, size=m)
        params_rand = np.concatenate([c_rand, log_roots_rand])
        initial_guesses.append(params_rand)
    
    # 4. Different root distributions
    for base in [1.5, 2.0, 2.5]:
        for start in [1.0, 2.0, 4.0]:
            roots_test = start * (base ** np.arange(m))
            if np.all(roots_test > 1e-4) and np.all(roots_test < 1e4):
                coeffs_desc = np.poly(roots_test)
                p_asc = coeffs_desc[::-1]
                c_test = A_inv @ p_asc
                log_roots_test = np.log(roots_test)
                params_test = np.concatenate([c_test, log_roots_test])
                initial_guesses.append(params_test)

    for i, init in enumerate(initial_guesses):
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
            
        try:
            result = minimize(
                cost_func, init, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
                best_roots = np.exp(best_params[m + 1:])
                
        except Exception:
            pass

    # If we have time, try DE runs
    if time.time() - start_time < deadline - 90:
        de_seeds = [42, 123, 456]
        for seed in de_seeds:
            if time.time() - start_time > deadline - 30:
                break
            try:
                result_de = differential_evolution(
                    cost_func, 
                    bounds, 
                    seed=seed, 
                    maxiter=100, 
                    popsize=15, 
                    tol=1e-10, 
                    polish=True,
                    init='sobol'
                )
                if result_de.fun < best_val:
                    best_val = result_de.fun
                    best_params = result_de.x.copy()
                    best_roots = np.exp(best_params[m + 1:])
            except Exception:
                pass

    # Final polish
    if time.time() - start_time < deadline - 5 and best_params is not None:
        try:
            polish_result = minimize(
                cost_func, best_params, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-16, 'maxls': 100}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_params = polish_result.x.copy()
                best_roots = np.exp(best_params[m + 1:])
        except Exception:
            pass

    # Ensure roots are positive and return
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-10)
        return list(best_roots)
    
    # Absolute fallback: zeros of L_12^(1)(x)
    try:
        laguerre_roots = roots_genlaguerre(m, alpha)[0]
        laguerre_roots = np.sort(laguerre_roots)
        laguerre_roots = np.maximum(laguerre_roots, 1e-10)
        return list(laguerre_roots)
    except Exception:
        return [4.0 * (2.0 ** i) for i in range(12)]
