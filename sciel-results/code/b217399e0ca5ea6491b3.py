"""Refined Cohn-Goncalves Laguerre construction optimizer."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
import time

def _compute_c4_bound_fast(roots):
    """Compute the C4 upper bound for given double-root positions using Gauss-Laguerre quadrature."""
    roots = np.asarray(roots, dtype=np.float64)
    if len(roots) != 12:
        return np.inf
    if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
        return np.inf
    
    n_quad = 200
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Vectorized computation of P(x) = prod (x - r_i)^2
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

def _objective(roots):
    """Objective function for optimization."""
    return _compute_c4_bound_fast(roots)

def _rational_potential_to_roots(params):
    """
    Convert 4-parameter rational potential V(x) = (a1*x^2 + a2*x + a3) / (x + b)
    to the 12 critical points (roots) of V'(x).
    V'(x) = 0 gives a cubic equation in x (after clearing denominator),
    but we need 12 roots, so this is a constrained ansatz.
    
    Actually, let's use a different approach: parametrize roots as
    r_i = exp(mu + sigma * z_i) where z_i are fixed quantile points,
    but allow a 4th parameter to modulate the spacing.
    
    Alternative: Use a 4-parameter family:
    r_i = base * exp(spread * (i - 5.5)/5.5 * shape)
    where base, spread, shape are 3 params, plus a global shift.
    
    Let me use: r_i = exp(a + b * (i-5.5) + c * (i-5.5)^2 + d * (i-5.5)^3)
    This is a 4-parameter polynomial-in-index parametrization.
    """
    a, b, c, d = params
    i = np.arange(12)
    t = (i - 5.5) / 5.5  # Normalized index in [-1, 1]
    log_roots = a + b * t + c * t**2 + d * t**3
    roots = np.exp(log_roots)
    
    # Ensure all positive and reasonable
    if np.any(roots <= 1e-10) or np.any(roots > 1e6):
        return None
    if np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
        return None
    
    roots = np.sort(roots)
    return roots

def _rational_potential_objective(params):
    """Objective for 4-parameter rational potential search."""
    roots = _rational_potential_to_roots(params)
    if roots is None:
        return np.inf
    return _compute_c4_bound_fast(roots)

def _lognormal_to_roots(mu, sigma):
    """Convert log-normal parameters (mu, sigma) to 12 sorted root locations."""
    if sigma < 1e-6:
        sigma = 1e-6
    p = (np.arange(1, 13)) / 12.0
    z = norm.ppf(p)
    roots = np.exp(mu + sigma * z)
    return np.sort(roots)

def _lognormal_objective(params):
    """Objective for log-normal parameter optimization."""
    mu, sigma = params
    if sigma < 1e-6:
        sigma = 1e-6
    if mu < -5 or mu > 5:
        return np.inf
    roots = _lognormal_to_roots(mu, sigma)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf
    return _compute_c4_bound_fast(roots)

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    deadline = time.time() + 580  # Safety margin before 630s kill
    
    # Start from standard Laguerre roots
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    
    best_roots = x0.copy()
    best_score = _compute_c4_bound_fast(best_roots)
    
    rng = np.random.default_rng(42)
    
    # ============================================================
    # PHASE 1: 4-parameter polynomial-in-index search
    # r_i = exp(a + b*t_i + c*t_i^2 + d*t_i^3) where t_i = (i-5.5)/5.5
    # ============================================================
    
    # Grid search over 4 parameters
    a_grid = np.linspace(-2.0, 2.0, 30)
    b_grid = np.linspace(-2.0, 2.0, 25)
    c_grid = np.linspace(-1.5, 1.5, 20)
    d_grid = np.linspace(-1.0, 1.0, 15)
    
    best_rp_score = np.inf
    best_rp_params = (0.0, 0.0, 0.0, 0.0)
    
    for a in a_grid:
        if time.time() > deadline - 300:
            break
        for b in b_grid:
            if time.time() > deadline - 300:
                break
            for c in c_grid:
                if time.time() > deadline - 300:
                    break
                for d in d_grid:
                    score = _rational_potential_objective([a, b, c, d])
                    if score < best_rp_score:
                        best_rp_score = score
                        best_rp_params = (a, b, c, d)
    
    if best_rp_score < best_score:
        best_score = best_rp_score
        best_roots = _rational_potential_to_roots(best_rp_params)
    
    # Refine with Nelder-Mead in 4D
    if time.time() < deadline - 200:
        try:
            result_nm = minimize(
                _rational_potential_objective,
                best_rp_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if result_nm.fun < best_score:
                candidate_roots = _rational_potential_to_roots(result_nm.x)
                if candidate_roots is not None:
                    score = _compute_c4_bound_fast(candidate_roots)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate_roots
        except Exception:
            pass
    
    # ============================================================
    # PHASE 2: 2D log-normal search (complementary family)
    # ============================================================
    
    mu_grid = np.linspace(-3.0, 3.0, 50)
    sigma_grid = np.linspace(0.1, 2.5, 40)
    
    best_ln_score = np.inf
    best_ln_params = (0.0, 0.5)
    
    for mu in mu_grid:
        if time.time() > deadline - 250:
            break
        for sigma in sigma_grid:
            score = _lognormal_objective([mu, sigma])
            if score < best_ln_score:
                best_ln_score = score
                best_ln_params = (mu, sigma)
    
    if best_ln_score < best_score:
        best_score = best_ln_score
        best_roots = _lognormal_to_roots(*best_ln_params)
    
    # Refine log-normal with Nelder-Mead
    if time.time() < deadline - 150:
        try:
            result_ln = minimize(
                _lognormal_objective,
                best_ln_params,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if result_ln.fun < best_score:
                candidate_roots = _lognormal_to_roots(*result_ln.x)
                score = _compute_c4_bound_fast(candidate_roots)
                if score < best_score:
                    best_score = score
                    best_roots = candidate_roots
        except Exception:
            pass
    
    # ============================================================
    # PHASE 3: Direct 12D optimization from multiple initial points
    # ============================================================
    
    initial_points = [x0.copy(), best_roots.copy()]
    
    # Perturbed Laguerre roots
    for scale in [0.1, 0.2, 0.3, 0.5]:
        perturbed = x0 * (1 + scale * rng.standard_normal(12))
        perturbed = np.maximum(perturbed, 1e-3)
        perturbed = np.sort(perturbed)
        initial_points.append(perturbed)
    
    # Log-uniform spacing
    for log_start in [-2.0, -1.5, -1.0, -0.5, 0.0]:
        for log_end in [1.0, 1.5, 2.0, 2.5, 3.0]:
            if log_end > log_start:
                log_roots = np.linspace(log_start, log_end, 12)
                roots = np.exp(log_roots)
                if roots[0] > 1e-3 and roots[-1] < 100:
                    initial_points.append(roots.copy())
    
    # Geometric spacing
    for start in [0.3, 0.5, 1.0]:
        for ratio in [1.3, 1.5, 1.8, 2.0]:
            geom = start * (ratio ** np.arange(12))
            if geom[-1] < 100 and geom[0] > 1e-3:
                initial_points.append(geom.copy())
    
    bounds = [(1e-6, 200.0)] * 12
    
    for x_init in initial_points:
        if time.time() > deadline - 100:
            break
        
        try:
            result = minimize(
                _objective,
                x_init,
                method='L-BFGS-B',
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
    
    # Differential evolution in 12D
    de_configs = [
        {'bounds': [(0.01, 50)] * 12, 'maxiter': 50, 'popsize': 15, 'seed': 42},
        {'bounds': [(0.05, 30)] * 12, 'maxiter': 45, 'popsize': 15, 'seed': 123},
        {'bounds': [(0.1, 20)] * 12, 'maxiter': 40, 'popsize': 18, 'seed': 777},
    ]
    
    for cfg in de_configs:
        if time.time() > deadline - 60:
            break
        try:
            result_de = differential_evolution(
                _objective,
                bounds=cfg['bounds'],
                maxiter=cfg['maxiter'],
                popsize=cfg['popsize'],
                tol=1e-10,
                seed=cfg['seed'],
                polish=False,
                updating='immediate',
                workers=1
            )
            
            if result_de.fun < best_score:
                candidate = np.sort(result_de.x)
                score = _compute_c4_bound_fast(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
            
            # Polish the DE result
            if np.all(result_de.x > 0):
                result_polish = minimize(
                    _objective,
                    result_de.x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-13}
                )
                if np.all(result_polish.x > 0) and not np.any(np.isnan(result_polish.x)):
                    candidate = np.sort(result_polish.x)
                    score = _compute_c4_bound_fast(candidate)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate.copy()
                        
        except Exception:
            pass
    
    # Final high-precision polish on best roots
    if time.time() < deadline - 10:
        try:
            for ftol, gtol, maxit in [(1e-14, 1e-12, 500), (1e-15, 1e-13, 400), (1e-16, 1e-14, 300)]:
                if time.time() > deadline - 5:
                    break
                result_final = minimize(
                    _objective,
                    best_roots,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': maxit, 'ftol': ftol, 'gtol': gtol}
                )
                if np.all(result_final.x > 0) and not np.any(np.isnan(result_final.x)):
                    candidate = np.sort(result_final.x)
                    score = _compute_c4_bound_fast(candidate)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate.copy()
        except Exception:
            pass
    
    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
