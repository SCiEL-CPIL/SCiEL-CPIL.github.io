"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize
import time

def _compute_c4_bound_fast(roots):
    """
    Compute the C4 upper bound for given double-root positions.
    Uses Gauss-Laguerre quadrature for numerical integration.
    Metric: int(P^2 * w) / (int(P * w))^2
    """
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf
    
    # Use Gauss-Laguerre quadrature (includes e^{-x} weight)
    n_quad = 200
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # P(x) = prod (x - r_i)^2
    # Compute log_P for numerical stability
    diff = x_quad[:, np.newaxis] - roots[np.newaxis, :]
    log_abs_diff = np.log(np.abs(diff) + 1e-300)
    log_P = 2.0 * np.sum(log_abs_diff, axis=1)
    
    P = np.exp(log_P)
    P2 = P * P
    
    int_P = np.sum(w_quad * P)
    int_P2 = np.sum(w_quad * P2)
    
    if int_P <= 0 or np.isinf(int_P) or np.isinf(int_P2):
        return np.inf
    
    c4 = int_P2 / (int_P * int_P)
    return c4

def _compute_c4_bound_grad(roots):
    """
    Compute the C4 upper bound and its gradient using Gauss-Laguerre quadrature.
    Returns (c4, grad) where grad has shape (12,).
    """
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf, np.zeros(12)
    
    n_quad = 200
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # P(x) = prod (x - r_i)^2
    diff = x_quad[:, np.newaxis] - roots[np.newaxis, :]
    log_abs_diff = np.log(np.abs(diff) + 1e-300)
    log_P = 2.0 * np.sum(log_abs_diff, axis=1)
    
    P = np.exp(log_P)
    P2 = P * P
    
    int_P = np.sum(w_quad * P)
    int_P2 = np.sum(w_quad * P2)
    
    if int_P <= 0 or np.isinf(int_P) or np.isinf(int_P2):
        return np.inf, np.zeros(12)
    
    c4 = int_P2 / (int_P * int_P)
    
    # Gradient computation
    # d/d r_j log_P = sum_i 2 * (x - r_i) / (x - r_j) ... actually simpler:
    # P(x) = prod_k (x - r_k)^2
    # dP/dr_j = P * 2 * sum_k delta(k,j) * 1/(x - r_j) = P * 2/(x - r_j)
    # So dP/dr_j = 2 * P / (x - r_j)
    # 
    # dc4/dr_j = [2 * int_P * d(int_P2)/dr_j - 2 * int_P2 * d(int_P)/dr_j] / (int_P^3)
    # 
    # d(int_P)/dr_j = sum_q w_q * 2 * P(x_q) / (x_q - r_j)
    # d(int_P2)/dr_j = sum_q w_q * 2 * P(x_q)^2 * 2 / (x_q - r_j)
    #                 = 4 * sum_q w_q * P2(x_q) / (x_q - r_j)
    
    inv_diff = 1.0 / (x_quad[:, np.newaxis] - roots[np.newaxis, :])
    # Handle potential division by zero
    inv_diff = np.where(np.abs(x_quad[:, np.newaxis] - roots[np.newaxis, :]) < 1e-15, 0.0, inv_diff)
    
    d_int_P = 2.0 * np.sum(w_quad[:, np.newaxis] * P[:, np.newaxis] * inv_diff, axis=0)
    d_int_P2 = 4.0 * np.sum(w_quad[:, np.newaxis] * P2[:, np.newaxis] * inv_diff, axis=0)
    
    grad = (2.0 * int_P * d_int_P2 - 2.0 * int_P2 * d_int_P) / (int_P ** 3)
    
    return c4, grad

def _objective_with_grad(roots):
    """Objective function with gradient for L-BFGS-B."""
    val, grad = _compute_c4_bound_grad(roots)
    if np.isinf(val):
        return 1e10, np.zeros(12)
    return val, grad

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) that minimize the C4 upper bound.
    
    Strategy:
    1. Use standard Laguerre roots as a strong initial guess.
    2. Generate diverse initial points.
    3. Refine each with L-BFGS-B using analytic gradients.
    4. Return the best set of roots found within the time limit.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s kill
    
    # Initial guess: Zeros of Laguerre L_12(x)
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    
    best_roots = x0.copy()
    best_score = _compute_c4_bound_fast(best_roots)
    
    if time.time() > deadline - 120:
        return [float(xi) for xi in best_roots]
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial points
    initial_points = [x0.copy()]
    
    # 1. Perturbed Laguerre roots with different scales
    for scale in [0.05, 0.1, 0.2, 0.3, 0.5]:
        perturbed = x0 * (1 + scale * rng.standard_normal(12))
        perturbed = np.maximum(perturbed, 1e-3)
        perturbed = np.sort(perturbed)
        initial_points.append(perturbed)
        
    # 2. Scaled Laguerre roots
    for scale in [0.8, 1.2, 1.5, 2.0]:
        scaled = x0 * scale
        initial_points.append(scaled.copy())
        
    # 3. Random uniform in [0.1, 30]
    for _ in range(5):
        rand_roots = rng.uniform(0.1, 30, size=12)
        rand_roots = np.sort(rand_roots)
        initial_points.append(rand_roots)
        
    # 4. Geometric spacing variations
    for start in [0.5, 1.0, 2.0]:
        for ratio in [1.3, 1.5, 1.8]:
            geom = start * (ratio ** np.arange(12))
            if geom[-1] < 100 and geom[0] > 0.1:
                initial_points.append(geom.copy())
                
    # 5. Log-uniform spacing
    for log_start in [-2.0, -1.0, 0.0]:
        for log_end in [1.0, 2.0, 3.0]:
            if log_end > log_start:
                log_roots = np.linspace(log_start, log_end, 12)
                roots = np.exp(log_roots)
                if roots[0] > 1e-3 and roots[-1] < 100:
                    initial_points.append(roots)

    # Refine each initial point with L-BFGS-B using analytic gradients
    bounds = [(1e-6, None)] * 12
    
    for i, x_init in enumerate(initial_points):
        if time.time() > deadline - 60:
            break
        
        try:
            result = minimize(
                _objective_with_grad,
                x_init,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            candidate = result.x
            if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                candidate = np.sort(candidate)
                score = _compute_c4_bound_fast(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
        except Exception:
            pass
    
    # Final polish on the best roots with tighter tolerance and analytic gradient
    if time.time() < deadline - 10:
        try:
            result_final = minimize(
                _objective_with_grad,
                best_roots,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if np.all(result_final.x > 0) and not np.any(np.isnan(result_final.x)):
                candidate = np.sort(result_final.x)
                score = _compute_c4_bound_fast(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
        except Exception:
            pass
    
    # Ensure the output is a list of 12 positive floats
    result = [max(float(r), 1e-6) for r in best_roots]
    if len(result) != 12:
        # Fallback to Laguerre roots if something went wrong
        result = [float(xi) for xi in x0]
        
    return result
# EVOLVE-BLOCK-END
