"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize
import time

# Precompute Gauss-Laguerre quadrature nodes
_N_QUAD = 200
_X_QUAD, _W_QUAD = np.polynomial.laguerre.laggauss(_N_QUAD)

def _compute_c4_and_grad(roots):
    """
    Compute the C4 upper bound and its analytical gradient for given double-root positions.
    C4 = int(x^2 * P^2 * w) / (int(x * P * w))^2
    where P(x) = prod_i (x - r_i)^2 and w(x) = e^{-x}.
    """
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf, None
    
    x = _X_QUAD
    w = _W_QUAD
    
    # P(x) = prod (x - r_i)^2
    # diff has shape (n_quad, 12)
    diff = x[:, np.newaxis] - roots[np.newaxis, :]
    
    # Compute log|diff| for numerical stability
    log_abs_diff = np.log(np.abs(diff) + 1e-300)
    log_P = 2.0 * np.sum(log_abs_diff, axis=1)
    
    P = np.exp(log_P)
    P2 = P * P
    
    # Compute integrals
    int_xP = np.sum(w * x * P)
    int_x2P2 = np.sum(w * x**2 * P2)
    
    if int_xP <= 0 or np.isinf(int_xP) or np.isinf(int_x2P2):
        return np.inf, None
    
    c4 = int_x2P2 / (int_xP * int_xP)
    
    if not np.isfinite(c4):
        return np.inf, None
    
    # Compute gradient analytically
    # dC4/dr_j = 2 * d(log P)/dr_j related terms
    # dP/dr_j = P * 2 * sum over k != j of 1/(x - r_k) ... actually for each root
    # For P = prod (x - r_i)^2, dP/dr_j = P * 2/(x - r_j)
    
    # Let's compute d(int_xP)/dr_j and d(int_x2P2)/dr_j
    # dP/dr_j = P * 2/(x - r_j)
    
    grad = np.zeros(12)
    
    # Precompute 1/(x - r_j) for each j
    # inv_diff[j, k] = 1/(x_k - r_j)
    inv_diff = 1.0 / (x[:, np.newaxis] - roots[np.newaxis, :])
    
    # dP/dr_j = P * 2 * inv_diff[:, j]
    # d(int_xP)/dr_j = sum(w * x * P * 2 * inv_diff[:, j])
    d_int_xP = 2.0 * np.sum(w[:, np.newaxis] * x[:, np.newaxis] * P[:, np.newaxis] * inv_diff, axis=0)
    
    # d(int_x2P2)/dr_j = sum(w * x^2 * 2 * P^2 * 2 * inv_diff[:, j]) = 4 * sum(w * x^2 * P^2 * inv_diff[:, j])
    d_int_x2P2 = 4.0 * np.sum(w[:, np.newaxis] * (x**2)[:, np.newaxis] * P2[:, np.newaxis] * inv_diff, axis=0)
    
    # C4 = A / B^2 where A = int_x2P2, B = int_xP
    # dC4/dr_j = (dA/dr_j * B^2 - A * 2 * B * dB/dr_j) / B^4
    #          = dA/dr_j / B^2 - 2 * A * dB/dr_j / B^3
    #          = (dA/dr_j) / B^2 - 2 * C4 * dB/dr_j / B
    
    B = int_xP
    A = int_x2P2
    
    for j in range(12):
        dA = d_int_x2P2[j]
        dB = d_int_xP[j]
        grad[j] = dA / (B * B) - 2.0 * C4_val * dB / B if False else (dA / (B * B) - 2.0 * (A / (B * B)) * (dB / B))
    
    # Fix the gradient computation
    C4_val = A / (B * B)
    for j in range(12):
        dA = d_int_x2P2[j]
        dB = d_int_xP[j]
        grad[j] = dA / (B * B) - 2.0 * C4_val * dB / B
    
    return c4, grad

def _objective(roots):
    """Objective function for optimization."""
    val, _ = _compute_c4_and_grad(roots)
    return val

def _objective_grad(roots):
    """Objective function with gradient for optimization."""
    val, grad = _compute_c4_and_grad(roots)
    if grad is None:
        return val, np.zeros(12)
    return val, grad

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) that minimize the C4 upper bound.
    
    Strategy:
    1. Use standard Laguerre roots as a strong initial guess.
    2. Generate diverse initial points (perturbations, geometric, random).
    3. Refine each with L-BFGS-B local optimization using analytical gradients.
    4. Return the best set of roots found within the time limit.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s kill
    
    # Initial guess: Zeros of Laguerre L_12(x)
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    
    best_roots = x0.copy()
    best_score, _ = _compute_c4_and_grad(best_roots)
    
    if not np.isfinite(best_score):
        best_score = np.inf
    
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

    # Refine each initial point with L-BFGS-B using analytical gradients
    bounds = [(1e-6, None)] * 12
    
    for i, x_init in enumerate(initial_points):
        if time.time() > deadline - 60:
            break
        
        try:
            # Use L-BFGS-B with bounds to keep roots positive
            result = minimize(
                _objective_grad,
                x_init,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            candidate = result.x
            if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                candidate = np.sort(candidate)
                score, _ = _compute_c4_and_grad(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
        except Exception:
            pass
    
    # Final polish on the best roots with tighter tolerance
    if time.time() < deadline - 10:
        try:
            result_final = minimize(
                _objective_grad,
                best_roots,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if np.all(result_final.x > 0) and not np.any(np.isnan(result_final.x)):
                candidate = np.sort(result_final.x)
                score, _ = _compute_c4_and_grad(candidate)
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
