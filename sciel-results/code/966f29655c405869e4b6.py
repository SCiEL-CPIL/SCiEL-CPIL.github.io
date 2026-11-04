"""Refined uncertainty inequality - Cohn-Goncalves Laguerre construction optimizer."""
import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize, differential_evolution
import time
import math

def _compute_c4_bound(roots):
    """Compute the C4 upper bound for given double-root positions."""
    roots = np.asarray(roots, dtype=np.float64)
    if len(roots) != 12:
        return np.inf
    if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
        return np.inf
    
    # Use Laguerre basis for fast, stable evaluation
    # P(x) = prod (x - r_i)^2 = [prod (x - r_i)]^2
    # Q(x) = prod (x - r_i), degree 12
    # P(x) = Q(x)^2, degree 24
    
    # We need: C4 = <P, P> / <P, 1>^2 in L^2(e^{-x})
    # where <f, g> = int_0^inf f(x) g(x) e^{-x} dx
    
    # Build Q(x) coefficients (ascending)
    coeffs_desc = np.poly(roots)  # descending: [1, c1, ..., c12]
    q_asc = coeffs_desc[::-1].copy()  # ascending: [c0, c1, ..., c12]
    
    # P(x) = Q(x)^2, compute coefficients
    p_asc = np.convolve(q_asc, q_asc)  # ascending coefficients of Q^2
    
    # Convert to Laguerre basis L_k^{(1)}
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k,j] = (-1)^j * binom(k+1, j+1) / j!
    
    max_deg = len(p_asc)  # 25
    A = np.zeros((max_deg, max_deg))
    for k in range(max_deg):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # Solve for Laguerre coefficients: p_asc = A @ c, so c = A^{-1} @ p_asc
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.pinv(A)
    
    c_P = A_inv @ p_asc  # Laguerre coefficients of P
    
    # Norm squared: ||P||^2 = sum c_k^2 * (k+1) for alpha=1
    weights = np.arange(max_deg) + 1
    norm_sq_P = np.sum(c_P**2 * weights)
    
    # <P, 1> = int P(x) e^{-x} dx
    # 1 in Laguerre basis: L_0^{(1)}(x) = 1, so c_1 = [1, 0, 0, ...]
    # <L_k, L_0> = delta_k0 * (0+1) = delta_k0
    # So <P, 1> = c_P[0] * 1 = c_P[0]
    # Wait, that's not right. Let me reconsider.
    
    # Actually <L_k^{(1)}, L_l^{(1)}> = delta_kl * (k+1)
    # 1 = L_0^{(1)}, so <P, 1> = <P, L_0> = c_P[0] * (0+1) = c_P[0]
    
    int_P = c_P[0]
    
    if norm_sq_P <= 1e-15 or abs(int_P) <= 1e-15:
        return np.inf
    
    c4 = norm_sq_P / (int_P * int_P)
    return c4

def _compute_norm_sq_only(roots):
    """Compute only the norm squared for faster optimization."""
    roots = np.asarray(roots, dtype=np.float64)
    if len(roots) != 12:
        return -np.inf
    if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
        return -np.inf
    
    coeffs_desc = np.poly(roots)
    q_asc = coeffs_desc[::-1].copy()
    p_asc = np.convolve(q_asc, q_asc)
    
    max_deg = len(p_asc)
    
    # Precomputed A_inv is expensive to recompute, so we use a cached version
    # But for now, compute inline
    A = np.zeros((max_deg, max_deg))
    for k in range(max_deg):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return -np.inf
    
    c_P = A_inv @ p_asc
    weights = np.arange(max_deg) + 1
    norm_sq_P = np.sum(c_P**2 * weights)
    
    if norm_sq_P <= 1e-15:
        return -np.inf
    
    return norm_sq_P

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    start_time = time.time()
    DEADLINE = 630.0
    SAFETY_MARGIN = 30.0
    
    def time_ok():
        return (time.time() - start_time) < (DEADLINE - SAFETY_MARGIN)
    
    # Precompute the Laguerre transformation matrix once
    max_deg = 25  # degree of P = 2*12 = 24, so 25 coefficients
    A = np.zeros((max_deg, max_deg))
    for k in range(max_deg):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.pinv(A)
    
    weights = np.arange(max_deg) + 1
    
    def fast_c4(roots):
        """Fast C4 computation using precomputed A_inv."""
        roots = np.asarray(roots, dtype=np.float64)
        if len(roots) != 12:
            return np.inf
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return np.inf
        
        coeffs_desc = np.poly(roots)
        q_asc = coeffs_desc[::-1].copy()
        p_asc = np.convolve(q_asc, q_asc)
        
        # Pad if needed
        if len(p_asc) < max_deg:
            p_asc = np.pad(p_asc, (0, max_deg - len(p_asc)))
        
        c_P = A_inv @ p_asc
        norm_sq_P = np.sum(c_P**2 * weights)
        int_P = c_P[0]
        
        if norm_sq_P <= 1e-15 or abs(int_P) <= 1e-15:
            return np.inf
        
        return norm_sq_P / (int_P * int_P)
    
    def objective_log(log_roots):
        """Objective in log-space: minimize C4."""
        roots = np.exp(log_roots)
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
        val = fast_c4(roots)
        if np.isinf(val) or np.isnan(val):
            return 1e10
        return val
    
    # Initial guess: standard Laguerre roots
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    best_roots = x0.copy()
    best_score = fast_c4(best_roots)
    
    if not time_ok():
        return [float(xi) for xi in best_roots]
    
    # Log-space bounds
    # Roots should be positive; typical range based on Laguerre roots ~[0.5, 30]
    # Allow wider: [0.01, 10000] -> log: [-4.6, 9.2]
    log_bounds = [(-4.0, 10.0)] * 12
    
    # Multi-start L-BFGS-B from different initial points
    rng = np.random.default_rng(42)
    
    # Generate diverse initial points in log-space
    initial_log_points = []
    
    # Start from standard Laguerre roots in log-space
    initial_log_points.append(np.log(x0))
    
    # Perturbed versions
    for _ in range(8):
        perturbed = x0 * (1 + 0.3 * rng.standard_normal(12))
        perturbed = np.maximum(perturbed, 1e-3)
        perturbed = np.sort(perturbed)
        initial_log_points.append(np.log(perturbed))
    
    # Geometric sequences
    for base in [2.0, 3.0, 5.0]:
        for ratio in [1.2, 1.5, 2.0]:
            geo = base * (ratio ** np.arange(12))
            if geo[-1] < 1e6:
                initial_log_points.append(np.log(geo))
    
    # Random initial points
    for _ in range(5):
        log_r = rng.uniform(-2.0, 8.0, size=12)
        initial_log_points.append(log_r)
    
    # L-BFGS-B optimization from each starting point
    for i, x_init in enumerate(initial_log_points):
        if not time_ok():
            break
        
        try:
            result = minimize(
                objective_log,
                x_init,
                method='L-BFGS-B',
                bounds=log_bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if result.fun < best_score:
                candidate_roots = np.exp(result.x)
                if np.all(candidate_roots > 0) and not np.any(np.isnan(candidate_roots)):
                    candidate_roots = np.sort(candidate_roots)
                    score = fast_c4(candidate_roots)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate_roots.copy()
        except Exception:
            pass
    
    if not time_ok():
        return [float(xi) for xi in best_roots]
    
    # Differential Evolution for global search
    # Use the best found so far as a reference
    best_log = np.log(best_roots)
    
    # DE with multiple seeds
    de_seeds = [101, 202, 303, 404, 505]
    
    for seed in de_seeds:
        if not time_ok():
            break
        
        try:
            # Tighter bounds around the best solution found
            if best_score < 40:  # If we've made progress, tighten bounds
                center = best_log
                spread = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
                de_bounds = [(center[i] - spread[i], center[i] + spread[i]) for i in range(12)]
            else:
                de_bounds = log_bounds
            
            result = differential_evolution(
                objective_log,
                de_bounds,
                seed=seed,
                maxiter=100,
                popsize=15,
                tol=1e-10,
                polish=True,
                init='sobol'
            )
            
            if result.fun < best_score:
                candidate_roots = np.exp(result.x)
                if np.all(candidate_roots > 0) and not np.any(np.isnan(candidate_roots)):
                    candidate_roots = np.sort(candidate_roots)
                    score = fast_c4(candidate_roots)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate_roots.copy()
        except Exception:
            pass
    
    if not time_ok():
        return [float(xi) for xi in best_roots]
    
    # Final polish with very tight tolerance
    best_log = np.log(best_roots)
    try:
        result = minimize(
            objective_log,
            best_log,
            method='L-BFGS-B',
            bounds=log_bounds,
            options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14}
        )
        
        if result.fun < best_score:
            candidate_roots = np.exp(result.x)
            if np.all(candidate_roots > 0) and not np.any(np.isnan(candidate_roots)):
                candidate_roots = np.sort(candidate_roots)
                score = fast_c4(candidate_roots)
                if score < best_score:
                    best_score = score
                    best_roots = candidate_roots.copy()
    except Exception:
        pass
    
    # Ensure positive and return
    best_roots = np.maximum(best_roots, 1e-10)
    return [float(xi) for xi in best_roots]
