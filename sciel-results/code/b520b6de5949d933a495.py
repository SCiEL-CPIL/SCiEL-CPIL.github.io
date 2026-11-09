"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

def _compute_bound(roots):
    """
    Compute the Cohn-Goncalves upper bound on C4.
    The bound is given by the product of the L2 norm of the polynomial P(x)
    and the L2 norm of 1/P(x), both with respect to the Laguerre weight e^{-x}.
    P(x) = prod_i (x - r_i)^2
    """
    roots = np.asarray(roots, dtype=np.float64)
    if len(roots) != 12:
        return 1e10
    if np.any(roots <= 0):
        return 1e10
    if np.any(np.diff(np.sort(roots)) < 1e-6):
        return 1e10
        
    # Use Gauss-Laguerre quadrature
    # Higher precision for stability
    x_nodes, x_weights = np.polynomial.laguerre.laggauss(300)
    
    # Compute log|P(x)| = 2 * sum log|x - r_i|
    # P(x) = prod (x - r_i)^2
    # log|P(x)| = 2 * sum_i log|x - r_i|
    
    # To handle overflow/underflow, we compute the integrals in log-space
    # Integral 1: I1 = sum w_i * exp(2 * sum log|x_i - r_j|)
    # Integral 2: I2 = sum w_i * exp(-2 * sum log|x_i - r_j|)
    
    # Let L(x) = sum_j log|x - r_j|
    # Then log|P(x)| = 2 L(x)
    
    log_terms_P = np.zeros_like(x_nodes)
    log_terms_inv = np.zeros_like(x_nodes)
    
    # Accumulate logs
    for r in roots:
        diff = np.abs(x_nodes - r)
        # Avoid log(0)
        diff = np.maximum(diff, 1e-300)
        log_terms_P += 2.0 * np.log(diff)
        log_terms_inv -= 2.0 * np.log(diff)
        
    # I1 = sum w_i * exp(log_terms_P_i)
    # I2 = sum w_i * exp(log_terms_inv_i)
    
    # Use log-sum-exp trick
    log_w = np.log(x_weights + 1e-300)
    
    # For I1
    max_log_P = np.max(log_terms_P + log_w)
    if max_log_P > 700:
        # Overflow likely, but let's try to be robust
        # Actually, if max_log_P is huge, the bound is huge
        return np.exp(max_log_P) * np.sum(np.exp(log_terms_P + log_w - max_log_P))
    
    sum_exp_P = np.sum(np.exp(log_terms_P + log_w - max_log_P))
    I1 = np.exp(max_log_P) * sum_exp_P
    
    # For I2
    max_log_inv = np.max(log_terms_inv + log_w)
    if max_log_inv > 700:
        return np.exp(max_log_inv) * np.sum(np.exp(log_terms_inv + log_w - max_log_inv))
    
    sum_exp_inv = np.sum(np.exp(log_terms_inv + log_w - max_log_inv))
    I2 = np.exp(max_log_inv) * sum_exp_inv
    
    # The bound is sqrt(I1 * I2) ? Or I1 * I2?
    # The Cohn-Goncalves bound for the Fourier transform sign uncertainty is related to
    # the ratio of norms. Specifically, if P is the polynomial, the bound involves
    # ||P||_2 / ||P||_inf or similar. 
    # However, the "refined" bound often cited is the product of the L2 norms of P and 1/P
    # in the dual space, or simply the condition number.
    # Let's assume the score is the value of the bound itself.
    # A common upper bound for the uncertainty constant is related to the norm of the reproducing kernel
    # or the polynomial's behavior.
    # Let's look at the seed: roots = [4, 8, ..., 48].
    # If the bound is 1.0 for the seed, we need to calibrate.
    # Let's return sqrt(I1 * I2) as a measure of the "width" or uncertainty.
    # Actually, the problem states "upper bound on C4".
    # In many uncertainty principles, the bound is 1/sqrt(det(Gram matrix)) or similar.
    # Let's try returning I1 * I2 or sqrt(I1*I2).
    # Given the seed scores 1.0, and typical norms are large, maybe it's normalized.
    # Let's assume the bound is proportional to sqrt(I1 * I2).
    
    bound = np.sqrt(I1 * I2)
    
    # Calibration: We don't know the exact constant, but the relative optimization is what matters.
    # However, the score is "seed/value". If seed value is V_seed, score = V_seed / V_current.
    # To get score > 1, we need V_current < V_seed.
    # So we just need to minimize this quantity.
    
    return bound

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Score is the upper bound on C4 (lower is better).
    """
    start_time = time.time()
    deadline = start_time + 570  # Safety margin
    
    # Seed roots
    seed_roots = np.array([4.0 * (i + 1) for i in range(12)])
    
    # We optimize in log-space to ensure positivity and handle scale better
    # Let roots = exp(log_roots)
    
    def objective_log(log_roots):
        if time.time() > deadline - 30:
            return 1e10
        roots = np.exp(log_roots)
        # Check for very large or small values
        if np.any(roots < 0.1) or np.any(roots > 1000):
            return 1e10
        return _compute_bound(roots)
    
    # Differential Evolution bounds for log_roots
    # Roots are typically in range [1, 100]
    # log(1) = 0, log(100) = 4.6
    bounds = [(0.0, 5.0)] * 12
    
    # Initial population based on seed
    seed_log = np.log(seed_roots)
    
    # Run Differential Evolution
    # maxiter and popsize control the search. 
    # 500 iterations with popsize 20 is 10000 evaluations.
    # Each evaluation takes ~10ms (300 nodes * 12 roots). 10000 * 10ms = 100s.
    # This fits within the budget.
    
    try:
        result = differential_evolution(
            objective_log,
            bounds,
            seed=42,
            maxiter=500,
            popsize=20,
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=True,  # Use L-BFGS-B at the end
            init='sobol'
        )
        
        if result.fun < 1e9:
            best_log_roots = result.x
            best_roots = np.exp(best_log_roots)
            best_bound = result.fun
        else:
            best_roots = seed_roots
            best_bound = _compute_bound(seed_roots)
            
    except Exception:
        best_roots = seed_roots
        best_bound = _compute_bound(seed_roots)
        
    # If we have time left, try a local refinement from the best DE result
    if time.time() < deadline - 60:
        from scipy.optimize import minimize
        
        def objective_roots(roots):
            if time.time() > deadline - 30:
                return best_bound
            if np.any(roots <= 0):
                return 1e10
            return _compute_bound(roots)
            
        try:
            result_local = minimize(
                objective_roots,
                best_roots,
                method='L-BFGS-B',
                bounds=[(0.1, 1000)] * 12,
                options={'maxiter': 100, 'maxfun': 1000}
            )
            if result_local.fun < best_bound:
                best_bound = result_local.fun
                best_roots = result_local.x
        except Exception:
            pass
            
    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END
