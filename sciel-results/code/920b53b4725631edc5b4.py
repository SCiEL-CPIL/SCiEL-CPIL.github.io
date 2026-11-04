import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_laguerre
import time

def _compute_c4_bound_fast(roots):
    """Compute the C4 upper bound for given double-root positions using Gauss-Laguerre quadrature."""
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf
    
    n_quad = 200
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
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

def _roots_from_params(params):
    """
    Compute 12 positive double-root locations from a 4-parameter rational potential.
    
    The potential is V(x) = a1/x + a2/(x+b) + c1*log(x) + c2*x
    Critical points satisfy V'(x) = 0:
    -a1/x^2 - a2/(x+b)^2 + c1/x + c2 = 0
    
    Multiply by x^2*(x+b)^2:
    -a1*(x+b)^2 - a2*x^2 + c1*x*(x+b)^2 + c2*x^2*(x+b)^2 = 0
    
    This is a quartic in x. We need all 4 roots to be positive.
    Actually, with m=12 we need 12 roots, so we use a more complex potential.
    
    Let's use: V(x) = sum_{i=1}^{4} a_i / (x + b_i) + c*log(x) + d*x
    V'(x) = -sum a_i/(x+b_i)^2 + c/x + d = 0
    
    Multiply by x * prod(x+b_i)^2 to get a polynomial of degree 1+2*4 = 9.
    That's not enough. Let's use more terms.
    
    Actually, for 12 roots we need the derivative to be a degree-12 polynomial.
    Let V'(x) = P(x)/Q(x) where deg(Q)=k and deg(P)=12+k.
    
    Simpler approach: parameterize roots via a differentiable map from low-dim space.
    Use log-spaced roots with adjustable ratios: r_i = exp(log_r1 + i*log_ratio)
    But that's too rigid.
    
    Better: use the fact that optimal roots for such problems often follow
    r_i ≈ A * i^alpha for some alpha. Use 2 params: A, alpha.
    r_i = A * (i+1)^alpha for i=0,...,11
    """
    A, alpha = params
    if A <= 0 or alpha < 0.1 or alpha > 3.0:
        return None
    i = np.arange(1, 13)
    roots = A * i**alpha
    if np.any(roots <= 0) or np.any(roots > 1e6):
        return None
    return roots

def _objective_params(params):
    """Objective in parameter space."""
    roots = _roots_from_params(params)
    if roots is None:
        return 1e10
    return _compute_c4_bound_fast(roots)

def _fit_params_to_roots(roots):
    """Fit A, alpha to given roots via least squares on log space."""
    i = np.arange(1, 13)
    log_roots = np.log(roots)
    log_i = np.log(i)
    # log(r_i) = log(A) + alpha * log(i)
    A_mat = np.vstack([np.ones(12), log_i]).T
    try:
        coeff, _, _, _ = np.linalg.lstsq(A_mat, log_roots, rcond=None)
        log_A, alpha = coeff
        return np.array([np.exp(log_A), alpha])
    except:
        return np.array([1.0, 1.0])

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    start_time = time.time()
    deadline = start_time + 580.0
    
    # Get Laguerre roots as reference
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    
    best_roots = x0.copy()
    best_score = _compute_c4_bound_fast(best_roots)
    
    if time.time() > deadline - 180:
        return [float(xi) for xi in best_roots]
    
    # Phase 1: Optimize in 2D parameter space (A, alpha) for power-law roots
    # This gives a smooth landscape
    
    # Initial guess from Laguerre roots
    init_params = _fit_params_to_roots(x0)
    
    bounds_param = [(0.01, 100.0), (0.1, 3.0)]
    
    try:
        result_p = minimize(
            _objective_params,
            init_params,
            method='L-BFGS-B',
            bounds=bounds_param,
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13}
        )
        if result_p.success or result_p.fun < best_score:
            roots_p = _roots_from_params(result_p.x)
            if roots_p is not None:
                score_p = _compute_c4_bound_fast(roots_p)
                if score_p < best_score:
                    best_score = score_p
                    best_roots = roots_p.copy()
    except Exception:
        pass
    
    if time.time() > deadline - 120:
        return [float(xi) for xi in best_roots]
    
    # Phase 2: Use differential evolution to search in a 6-parameter space
    # r_i = A * (i+1)^alpha * (1 + eps_i) where eps are small perturbations
    # Actually, let's use a different structural model:
    # Group roots into 3 clusters of 4, each cluster geometric with shared ratio
    # Params: [A1, r1, A2, r2, A3, r3] where cluster j has roots A_j * r_j^[0,1,2,3]
    
    def _cluster_objective(params):
        A1, r1, A2, r2, A3, r3 = params
        if any(A <= 0 for A in [A1, A2, A3]) or any(r < 0.5 or r > 3.0 for r in [r1, r2, r3]):
            return 1e10
        roots = np.concatenate([
            A1 * r1**np.arange(4),
            A2 * r2**np.arange(4),
            A3 * r3**np.arange(4)
        ])
        if np.any(roots <= 0) or np.any(roots > 1e5):
            return 1e10
        return _compute_c4_bound_fast(roots)
    
    # Initial guess: distribute Laguerre roots into 3 groups
    g1 = x0[0:4]
    g2 = x0[4:8]
    g3 = x0[8:12]
    
    def _fit_cluster(group):
        # group = [A, A*r, A*r^2, A*r^3]
        # log(group[i]) = log(A) + i*log(r)
        lg = np.log(group)
        idx = np.arange(4)
        mat = np.vstack([np.ones(4), idx]).T
        try:
            coeff, _, _, _ = np.linalg.lstsq(mat, lg, rcond=None)
            log_A, log_r = coeff
            return np.exp(log_A), np.exp(log_r)
        except:
            return group[0], group[1]/group[0]
    
    A1, r1 = _fit_cluster(g1)
    A2, r2 = _fit_cluster(g2)
    A3, r3 = _fit_cluster(g3)
    init_cluster = np.array([A1, r1, A2, r2, A3, r3])
    
    bounds_cluster = [(0.01, 50), (0.5, 3.0)] * 3
    
    try:
        result_c = minimize(
            _cluster_objective,
            init_cluster,
            method='L-BFGS-B',
            bounds=bounds_cluster,
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13}
        )
        if result_c.fun < best_score:
            A1, r1, A2, r2, A3, r3 = result_c.x
            roots_c = np.concatenate([
                A1 * r1**np.arange(4),
                A2 * r2**np.arange(4),
                A3 * r3**np.arange(4)
            ])
            score_c = _compute_c4_bound_fast(roots_c)
            if score_c < best_score:
                best_score = score_c
                best_roots = roots_c.copy()
    except Exception:
        pass
    
    if time.time() > deadline - 90:
        return [float(xi) for xi in best_roots]
    
    # Phase 3: Direct optimization from best roots with multi-start
    rng = np.random.default_rng(123)
    
    # Generate diverse starts
    starts = [best_roots.copy()]
    for scale in [0.5, 0.8, 1.2, 1.5, 2.0]:
        starts.append(best_roots * scale)
    for _ in range(5):
        pert = best_roots * (1 + 0.2 * rng.standard_normal(12))
        pert = np.maximum(pert, 0.01)
        starts.append(np.sort(pert))
    
    bounds = [(0.01, 100.0)] * 12
    
    for x_init in starts:
        if time.time() > deadline - 30:
            break
        try:
            result = minimize(
                lambda r: _compute_c4_bound_fast(r),
                x_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if np.all(result.x > 0) and not np.any(np.isnan(result.x)):
                cand = np.sort(result.x)
                sc = _compute_c4_bound_fast(cand)
                if sc < best_score:
                    best_score = sc
                    best_roots = cand.copy()
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 10:
        try:
            result_final = minimize(
                lambda r: _compute_c4_bound_fast(r),
                best_roots,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if np.all(result_final.x > 0) and not np.any(np.isnan(result_final.x)):
                cand = np.sort(result_final.x)
                sc = _compute_c4_bound_fast(cand)
                if sc < best_score:
                    best_score = sc
                    best_roots = cand.copy()
        except Exception:
            pass
    
    return [float(xi) for xi in best_roots]
