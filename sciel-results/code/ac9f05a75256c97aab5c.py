import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_laguerre
import time
import math

def _compute_c4_bound_laguerre(roots):
    """Compute the C4 upper bound using Laguerre basis norm.
    
    For P(x) = prod(x - r_i) (degree 12), the C4 bound is related to
    1 / ||P||^2 where ||.|| is the Laguerre inner product with alpha=1.
    ||P||^2 = sum_k c_k^2 * (k+1) where P = sum c_k L_k^{(1)}.
    """
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
        return np.inf
    
    # Build polynomial coefficients in standard basis (ascending)
    # P(x) = prod(x - r_i) = x^12 - e1*x^11 + e2*x^10 - ... + e12
    # np.poly returns descending: [1, -e1, e2, ..., (-1)^12*e12]
    coeffs_desc = np.poly(roots)
    p = coeffs_desc[::-1]  # ascending: [e12*(-1)^12, ..., -e1, 1]
    
    # Transform to Laguerre basis L_k^{(1)}
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # P(x) = sum_j p_j x^j = sum_k c_k L_k^{(1)}(x) = sum_j (sum_k c_k A[k,j]) x^j
    # So p_j = sum_k c_k A[k,j] => p = A^T c => c = (A^T)^{-1} p
    # A is upper triangular (A[k,j] = 0 for j > k)
    
    # Precompute A and its inverse once (cached)
    if not hasattr(_compute_c4_bound_laguerre, '_A_T_inv'):
        A = np.zeros((13, 13))
        for k in range(13):
            for j in range(k + 1):
                A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
        try:
            _compute_c4_bound_laguerre._A_T_inv = np.linalg.inv(A.T)
        except np.linalg.LinAlgError:
            _compute_c4_bound_laguerre._A_T_inv = np.linalg.pinv(A.T)
    
    c = _compute_c4_bound_laguerre._A_T_inv @ p
    
    # Norm squared: <L_k, L_l> = delta_kl * (k+1) for alpha=1
    norm_sq = np.sum(c**2 * (np.arange(13) + 1))
    
    if norm_sq <= 1e-10 or np.isinf(norm_sq):
        return np.inf
    
    return 1.0 / norm_sq

def _objective_log(log_roots):
    """Objective in log-space for unconstrained optimization."""
    roots = np.exp(log_roots)
    if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
        return 1e10
    return _compute_c4_bound_laguerre(roots)

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s kill
    
    def time_ok():
        return (time.time() - start_time) < (deadline - 30.0)
    
    # Get standard Laguerre roots of degree 12 as initial guess
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    
    # Convert to log-space
    log_x0 = np.log(x0)
    
    best_roots = x0.copy()
    best_score = _compute_c4_bound_laguerre(best_roots)
    
    if not time_ok():
        return [float(xi) for xi in best_roots]
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial points in log-space
    initial_points = [log_x0.copy()]
    
    # Perturbed Laguerre roots with different scales
    for scale in [0.05, 0.1, 0.2, 0.3, 0.5]:
        perturbed_log = log_x0 + scale * rng.standard_normal(12)
        initial_points.append(perturbed_log)
    
    # Scaled Laguerre roots
    for scale in [0.5, 0.8, 1.2, 1.5, 2.0, 3.0]:
        scaled = x0 * scale
        scaled = np.maximum(scaled, 1e-4)
        initial_points.append(np.log(scaled))
    
    # Random uniform in log-space
    for _ in range(5):
        rand_log = rng.uniform(-2.0, 8.0, size=12)
        initial_points.append(rand_log)
    
    # Geometric spacing variations in log-space
    for start_log in [math.log(0.5), math.log(1.0), math.log(2.0)]:
        for ratio_log in [math.log(1.3), math.log(1.5), math.log(1.8)]:
            geom_log = start_log + np.arange(12) * ratio_log
            if geom_log[0] > -5 and geom_log[-1] < 12:
                initial_points.append(geom_log)
    
    # Try L-BFGS-B from each initial point in log-space
    bounds_log = [(-5.0, 12.0)] * 12
    
    for x_init in initial_points:
        if not time_ok():
            break
        
        try:
            result = minimize(
                _objective_log,
                x_init,
                method='L-BFGS-B',
                bounds=bounds_log,
                options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            
            candidate_log = result.x
            candidate = np.exp(candidate_log)
            if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                candidate = np.sort(candidate)
                score = _compute_c4_bound_laguerre(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
        except Exception:
            pass
    
    # Run a few differential evolution runs for global search
    if time_ok():
        de_bounds = [(-4.0, 10.0)] * 12
        for seed in [42, 123, 456]:
            if not time_ok():
                break
            try:
                result_de = differential_evolution(
                    _objective_log,
                    de_bounds,
                    seed=seed,
                    maxiter=80,
                    popsize=15,
                    tol=1e-10,
                    polish=True,
                    init='sobol'
                )
                if result_de.fun < best_score:
                    candidate = np.exp(result_de.x)
                    if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                        candidate = np.sort(candidate)
                        score = _compute_c4_bound_laguerre(candidate)
                        if score < best_score:
                            best_score = score
                            best_roots = candidate.copy()
            except Exception:
                pass
    
    # Final high-precision polish on best roots
    if time_ok():
        try:
            best_log = np.log(best_roots)
            result_final = minimize(
                _objective_log,
                best_log,
                method='L-BFGS-B',
                bounds=[(-10.0, 15.0)] * 12,
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-16}
            )
            if np.all(result_final.x > -10) and np.all(result_final.x < 15):
                candidate = np.exp(result_final.x)
                if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                    candidate = np.sort(candidate)
                    score = _compute_c4_bound_laguerre(candidate)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate.copy()
        except Exception:
            pass
    
    return [float(xi) for xi in best_roots]
