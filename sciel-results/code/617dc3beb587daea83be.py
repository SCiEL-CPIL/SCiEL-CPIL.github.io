"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize
import time

# Precompute Gauss-Laguerre quadrature points
# N=200 is usually sufficient for high precision with 12 roots
N_QUAD = 200
x_quad, w_quad = np.polynomial.laguerre.laggauss(N_QUAD)
# Precompute e^{-x} for quadrature points
exp_neg_x = np.exp(-x_quad)
w_exp = w_quad * exp_neg_x

def _compute_c4_and_grad(roots):
    """
    Compute C4 and its gradient with respect to roots.
    roots: array of shape (12,)
    Returns: (c4, grad)
    """
    roots = np.asarray(roots, dtype=np.float64)
    m = len(roots)
    
    # Check validity
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf, np.zeros(m)
    
    # Compute P(x) at quadrature points
    # P(x) = prod (x - x_i)^2
    # log_P for stability? P can be large. 
    # Let's check magnitude. Roots ~ 1-20. x_quad ~ 0-100.
    # Max P ~ 100^24 ~ 1e48. P^2 ~ 1e96. Fits in float64 (max ~ 1e308).
    
    # Vectorized computation of P
    # diff shape: (N_QUAD, m)
    diff = x_quad[:, np.newaxis] - roots[np.newaxis, :]
    
    # Calculate log_P to avoid overflow/underflow in intermediate steps if needed, 
    # but direct product is likely fine for m=12.
    # Let's use log-sum-exp style if we were summing, but here it's product.
    # log_P = 2 * sum(log(|x_q - x_i|))
    # However, we need P and P^2.
    # If we use logs:
    # log_P = 2 * sum(log(abs(diff)), axis=1)
    # P = exp(log_P)
    # P2 = exp(2*log_P)
    
    # Direct multiplication might be faster and accurate enough for m=12.
    # Let's try direct first.
    P = np.ones(N_QUAD)
    for i in range(m):
        P *= (x_quad - roots[i])**2
        
    P2 = P * P
    
    int_P = np.sum(w_exp * P)
    int_P2 = np.sum(w_exp * P2)
    
    if int_P <= 0 or np.isinf(int_P) or np.isinf(int_P2):
        return np.inf, np.zeros(m)
        
    c4 = int_P2 / (int_P * int_P)
    
    # Compute Gradient
    # dI0/dx_k = 2 * sum( w_exp * P(x_q) / (x_q - x_k) )
    # dI1/dx_k = 4 * sum( w_exp * P(x_q)^2 / (x_q - x_k) )
    
    grad = np.zeros(m)
    
    # To speed up, we can compute the sums for all k at once?
    # diff is (N_QUAD, m).
    # inv_diff = 1 / diff. Careful with zeros.
    # Since P(x_k) = 0, the term P(x_q)/(x_q-x_k) is 0/0 if x_q==x_k.
    # But as established, the limit is 0.
    # We can mask where diff is close to 0 and set contribution to 0.
    
    # Actually, P is already computed.
    # term0 = w_exp * P  # shape (N_QUAD,)
    # term1 = w_exp * P2 # shape (N_QUAD,)
    
    # For each k, we need sum_q term0[q] / (x_q - x_k)
    # This is a matrix multiplication if we construct the inverse diff matrix.
    # inv_diff = 1.0 / diff  # (N_QUAD, m)
    # Handle singularities: if |x_q - x_k| < eps, set inv to 0.
    
    eps = 1e-10
    mask = np.abs(diff) < eps
    inv_diff = np.zeros_like(diff)
    np.divide(1.0, diff, out=inv_diff, where=~mask)
    
    # dI0/dx_k = 2 * sum_q (w_exp[q] * P[q] * inv_diff[q, k])
    # = 2 * (w_exp * P) @ inv_diff
    dI0 = 2.0 * ((w_exp * P) @ inv_diff)
    
    # dI1/dx_k = 4 * sum_q (w_exp[q] * P2[q] * inv_diff[q, k])
    # = 4 * (w_exp * P2) @ inv_diff
    dI1 = 4.0 * ((w_exp * P2) @ inv_diff)
    
    # dC4/dx_k = dI1/dx_k * (1/int_P^2) - 2 * I1 * dI0/dx_k * (1/int_P^3)
    # = (dI1 * int_P - 2 * I1 * dI0) / int_P^3
    
    denom = int_P**3
    grad = (dI1 * int_P - 2.0 * int_P2 * dI0) / denom
    
    return c4, grad

def _objective_and_grad(y):
    """
    Objective and gradient for L-BFGS-B.
    y: log(roots)
    """
    roots = np.exp(y)
    c4, grad_roots = _compute_c4_and_grad(roots)
    
    if np.isinf(c4):
        return c4, np.zeros_like(y)
        
    # Chain rule: dy/dx = 1/x
    grad_y = grad_roots / roots
    return c4, grad_y

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    deadline = time.time() + 580  # Safety margin
    
    # Start from standard Laguerre roots
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    
    best_roots = x0.copy()
    best_score, _ = _compute_c4_and_grad(best_roots)
    
    # Prepare initial points in log-space
    initial_points = []
    
    # 1. Standard Laguerre roots
    initial_points.append(np.log(x0))
    
    # 2. Scaled Laguerre roots (spread out)
    for scale in [1.2, 1.5, 2.0, 0.8]:
        scaled = x0 * scale
        initial_points.append(np.log(scaled))
        
    # 3. Shifted? No, must be positive.
    
    # 4. Random perturbations in log-space
    rng = np.random.default_rng(42)
    for _ in range(10):
        noise = rng.normal(0, 0.5, 12)
        perturbed_log = np.log(x0) + noise
        initial_points.append(perturbed_log)
        
    # 5. Geometric progression
    for start_log in [-1.0, 0.0, 1.0]:
        # Generate 12 points in log space, linearly spaced
        end_log = start_log + 3.0 # e^3 ~ 20
        geom_log = np.linspace(start_log, end_log, 12)
        initial_points.append(geom_log)

    bounds = [(np.log(1e-6), np.log(1000))] * 12
    
    # Optimization Loop
    for i, y_init in enumerate(initial_points):
        if time.time() > deadline - 30:
            break
            
        try:
            result = minimize(
                _objective_and_grad,
                y_init,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            
            if result.success and result.fun < best_score:
                y_opt = result.x
                x_opt = np.exp(y_opt)
                x_opt = np.sort(x_opt)
                # Verify
                score, _ = _compute_c4_and_grad(x_opt)
                if score < best_score:
                    best_score = score
                    best_roots = x_opt.copy()
                    
        except Exception as e:
            pass

    # Final high-precision polish
    if time.time() < deadline - 10:
        try:
            y_best = np.log(best_roots)
            result_final = minimize(
                _objective_and_grad,
                y_best,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if result_final.fun < best_score:
                y_opt = result_final.x
                x_opt = np.exp(y_opt)
                x_opt = np.sort(x_opt)
                score, _ = _compute_c4_and_grad(x_opt)
                if score < best_score:
                    best_score = score
                    best_roots = x_opt.copy()
        except Exception:
            pass

    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
