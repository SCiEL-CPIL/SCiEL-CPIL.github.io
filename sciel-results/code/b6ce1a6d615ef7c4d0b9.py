"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize
import time

def _compute_c4_bound(roots):
    """Compute the C4 upper bound for given double-root positions using Gauss-Laguerre quadrature."""
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf
    
    n_quad = 200
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # P(x) = prod (x - x_i)^2
    # Compute log_P for numerical stability
    log_P = np.zeros(n_quad)
    for r in roots:
        log_P += 2.0 * np.log(np.abs(x_quad - r) + 1e-300)
    
    P = np.exp(log_P)
    P2 = P * P
    
    int_P = np.sum(w_quad * P)
    int_P2 = np.sum(w_quad * P2)
    
    if int_P <= 0 or np.isinf(int_P) or np.isinf(int_P2):
        return np.inf
    
    c4 = int_P2 / (int_P * int_P)
    return c4

def _compute_c4_bound_fast(roots):
    """Faster version using vectorized operations where possible."""
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf
    
    n_quad = 200
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Vectorized: compute log_P for all roots at once
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

def _objective_from_log(log_roots):
    """Objective in log-space: roots = exp(log_roots)."""
    roots = np.exp(log_roots)
    return _compute_c4_bound_fast(roots)

def _geometric_objective(params):
    """Objective for geometric parameterization: roots = r0 * q^i for i=0..11."""
    r0_log, log_q = params
    r0 = np.exp(r0_log)
    q = np.exp(log_q)
    roots = r0 * (q ** np.arange(12))
    if np.any(roots <= 0) or np.any(roots > 1e6):
        return 1e10
    return _compute_c4_bound_fast(roots)

def _nelder_mead_minimize(func, x0, bounds, maxiter=500, tol=1e-12):
    """Nelder-Mead with bounds enforcement."""
    n = len(x0)
    # Create initial simplex
    simplex = [np.array(x0, dtype=np.float64)]
    for i in range(n):
        step = np.zeros(n)
        step[i] = 0.1 * max(abs(x0[i]), 1.0)
        simplex.append(np.array(x0 + step, dtype=np.float64))
    
    def _clamp(x):
        for i in range(n):
            if bounds[i][0] is not None and x[i] < bounds[i][0]:
                x[i] = bounds[i][0]
            if bounds[i][1] is not None and x[i] > bounds[i][1]:
                x[i] = bounds[i][1]
        return x
    
    for simplex_pt in simplex:
        simplex_pt[:] = _clamp(simplex_pt)
    
    for it in range(maxiter):
        # Evaluate
        vals = [func(pt) for pt in simplex]
        
        # Sort
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        
        # Check convergence
        if abs(vals[0] - vals[-1]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        alpha = 1.0
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = _clamp(x_r)
        f_r = func(x_r)
        
        if f_r < vals[0]:
            # Extension
            gamma = 2.0
            x_e = centroid + gamma * (centroid - simplex[-1])
            x_e = _clamp(x_e)
            f_e = func(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                vals[-1] = f_e
            else:
                simplex[-1] = x_r
                vals[-1] = f_r
        elif f_r < vals[-2]:
            simplex[-1] = x_r
            vals[-1] = f_r
        else:
            # Contraction
            rho = 0.5
            x_c = centroid + rho * (simplex[-1] - centroid)
            x_c = _clamp(x_c)
            f_c = func(x_c)
            if f_c < vals[-1]:
                simplex[-1] = x_c
                vals[-1] = f_c
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    simplex[i] = _clamp(simplex[i])
                vals = [func(pt) for pt in simplex]
    
    return simplex[0], vals[0]

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    deadline = time.time() + 590  # Safety margin before 630s kill
    
    # Start from standard Laguerre roots
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    
    best_roots = x0.copy()
    best_score = _compute_c4_bound_fast(best_roots)
    
    if time.time() > deadline - 120:
        return [float(xi) for xi in best_roots]
    
    # === Geometric parameterization search ===
    # The insight: optimal roots follow a geometric progression
    # roots_i = r0 * q^i for i = 0, 1, ..., 11
    # So we optimize over (log_r0, log_q)
    
    # Initial estimates for geometric parameters
    # From Laguerre roots, estimate ratio
    log_roots0 = np.log(x0)
    # Fit a line to log_roots vs index
    indices = np.arange(12)
    coeffs = np.polyfit(indices, log_roots0, 1)
    log_q_init = coeffs[0]
    log_r0_init = coeffs[1]
    
    # Multistart in log-space for geometric
    rng = np.random.default_rng(42)
    
    geo_starts = []
    # From Laguerre fit
    geo_starts.append([log_r0_init, log_q_init])
    
    # Perturbed
    for scale in [0.05, 0.1, 0.2, 0.3]:
        geo_starts.append([
            log_r0_init + scale * rng.standard_normal(),
            log_q_init + scale * rng.standard_normal()
        ])
    
    # Broader exploration
    for r0_log in [-2, -1, -0.5, 0, 0.5, 1.0, 1.5, 2.0]:
        for q_log in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
            geo_starts.append([r0_log, q_log])
    
    # L-BFGS-B in geometric log-space
    geo_bounds = [(-5, 5), (-2, 3)]
    
    for x_init in geo_starts:
        if time.time() > deadline - 60:
            break
        try:
            result = minimize(
                _geometric_objective,
                x_init,
                method='L-BFGS-B',
                bounds=geo_bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result.fun < best_score:
                r0 = np.exp(result.x[0])
                q = np.exp(result.x[1])
                candidate = r0 * (q ** np.arange(12))
                if np.all(candidate > 0) and np.all(candidate < 1e6):
                    candidate = np.sort(candidate)
                    score = _compute_c4_bound_fast(candidate)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate.copy()
        except Exception:
            pass
    
    # Nelder-Mead in geometric space for robustness
    if time.time() < deadline - 30:
        for x_init in geo_starts[:20]:
            if time.time() > deadline - 20:
                break
            try:
                x_opt, f_opt = _nelder_mead_minimize(
                    _geometric_objective, x_init, geo_bounds, maxiter=300
                )
                if f_opt < best_score:
                    r0 = np.exp(x_opt[0])
                    q = np.exp(x_opt[1])
                    candidate = r0 * (q ** np.arange(12))
                    if np.all(candidate > 0) and np.all(candidate < 1e6):
                        candidate = np.sort(candidate)
                        score = _compute_c4_bound_fast(candidate)
                        if score < best_score:
                            best_score = score
                            best_roots = candidate.copy()
            except Exception:
                pass
    
    # === Full 12D optimization from best geometric result ===
    # Use the geometric result as starting point for full optimization
    if time.time() < deadline - 30:
        # L-BFGS-B in log-space for all 12 roots
        log_bounds = [(-20, 20)] * 12
        log_roots_init = np.log(best_roots)
        
        try:
            result = minimize(
                _objective_from_log,
                log_roots_init,
                method='L-BFGS-B',
                bounds=log_bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            candidate = np.exp(result.x)
            if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                candidate = np.sort(candidate)
                score = _compute_c4_bound_fast(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
        except Exception:
            pass
        
        # Nelder-Mead in log-space
        if time.time() < deadline - 15:
            try:
                x_opt, f_opt = _nelder_mead_minimize(
                    _objective_from_log, log_roots_init, log_bounds, maxiter=400
                )
                candidate = np.exp(x_opt)
                if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                    candidate = np.sort(candidate)
                    score = _compute_c4_bound_fast(candidate)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate.copy()
            except Exception:
                pass
    
    # === Additional multistart in full log-space ===
    if time.time() < deadline - 30:
        # Generate diverse starting points
        full_starts = [np.log(best_roots)]
        
        # Perturbed best
        for scale in [0.05, 0.1, 0.2]:
            perturbed = np.log(best_roots) + scale * rng.standard_normal(12)
            full_starts.append(perturbed)
        
        # From Laguerre roots with perturbations
        for scale in [0.1, 0.2, 0.3, 0.5]:
            perturbed = np.log(x0) + scale * rng.standard_normal(12)
            full_starts.append(perturbed)
        
        # Random in log-space
        for _ in range(10):
            rand_log = rng.uniform(-3, 3, size=12)
            full_starts.append(rand_log)
        
        # Geometric with noise
        for q_log in [0.2, 0.3, 0.4, 0.5, 0.6]:
            for r0_log in [-1, 0, 1]:
                geom_log = r0_log + q_log * np.arange(12)
                noisy = geom_log + 0.1 * rng.standard_normal(12)
                full_starts.append(noisy)
        
        for x_init in full_starts:
            if time.time() > deadline - 20:
                break
            try:
                result = minimize(
                    _objective_from_log,
                    x_init,
                    method='L-BFGS-B',
                    bounds=log_bounds,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                candidate = np.exp(result.x)
                if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                    candidate = np.sort(candidate)
                    score = _compute_c4_bound_fast(candidate)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate.copy()
            except Exception:
                pass
    
    # === Final high-precision polish ===
    if time.time() < deadline - 5:
        try:
            log_roots_final = np.log(best_roots)
            result_final = minimize(
                _objective_from_log,
                log_roots_final,
                method='L-BFGS-B',
                bounds=log_bounds,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            candidate = np.exp(result_final.x)
            if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                candidate = np.sort(candidate)
                score = _compute_c4_bound_fast(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
        except Exception:
            pass
    
    # Ensure all roots are positive and finite
    best_roots = np.asarray(best_roots, dtype=np.float64)
    best_roots = np.maximum(best_roots, 1e-10)
    best_roots = np.sort(best_roots)
    
    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
