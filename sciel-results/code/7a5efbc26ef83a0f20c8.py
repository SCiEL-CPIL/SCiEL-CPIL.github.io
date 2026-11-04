# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import roots_genlaguerre, eval_genlaguerre

# Precompute factorials for moment computation
_FACTS = np.array([math.factorial(k) for k in range(100)], dtype=np.float64)

def _poly_moment(coeffs):
    """Compute integral of poly (ascending coeff order) * e^{-x} dx over [0,inf)."""
    n = len(coeffs)
    if n == 0:
        return 0.0
    m = min(n, 100)
    return float(np.dot(coeffs[:m], _FACTS[:m]))

def _build_p_coeffs(roots):
    """Build P(x) = prod (x - xi)^2 as coefficient array (ascending order)."""
    p_coeffs = np.array([1.0])
    for r in roots:
        new_len = len(p_coeffs) + 2
        new_coeffs = np.zeros(new_len)
        new_coeffs[2:] += p_coeffs
        new_coeffs[1:-1] += (-2 * r) * p_coeffs
        new_coeffs[:len(p_coeffs)] += (r * r) * p_coeffs
        p_coeffs = new_coeffs
    return p_coeffs

def _compute_c4(roots, need_grad=False):
    """Compute C4 = I2 / I1^2 where I1 = int P e^{-x} dx, I2 = int P^2 e^{-x} dx."""
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    n = len(roots)
    if n == 0:
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    # Build P(x) = prod (x - xi)^2
    p_coeffs = _build_p_coeffs(roots)
    
    # Build P2 = P^2
    p2_coeffs = np.convolve(p_coeffs, p_coeffs)
    
    I1 = _poly_moment(p_coeffs)
    I2 = _poly_moment(p2_coeffs)
    
    if I1 <= 0 or not np.isfinite(I1) or not np.isfinite(I2):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf
        
    c4 = I2 / (I1 * I1)
    if not need_grad:
        return float(c4)
    
    # Analytical gradient
    grad = np.zeros_like(roots)
    for i in range(n):
        # Compute R_i(x) = prod_{j != i} (x - xj)^2
        R_i = _build_p_coeffs(np.delete(roots, i))
        
        # Moments of R_i
        M0_Ri = _poly_moment(R_i)
        # x * R_i: shift coeffs by 1
        xR_i = np.zeros(len(R_i) + 1)
        xR_i[1:] = R_i
        M1_Ri = _poly_moment(xR_i)
        
        dI1_di = -4.0 * (M1_Ri - roots[i] * M0_Ri)
        
        # Compute R_i^2
        Ri_sq = np.convolve(R_i, R_i)
        
        # Moments of R_i^2
        M0_Ri2 = _poly_moment(Ri_sq)
        xRi_sq = np.zeros(len(Ri_sq) + 1)
        xRi_sq[1:] = Ri_sq
        M1_Ri2 = _poly_moment(xRi_sq)
        
        dI2_di = -8.0 * (M1_Ri2 - roots[i] * M0_Ri2)
        
        # dC4/dxi = dI2_di / I1^2 - 2 * I2 * dI1_di / I1^3
        grad[i] = dI2_di / (I1 * I1) - 2.0 * I2 * dI1_di / (I1 * I1 * I1)
        
    return float(c4), grad

def _objective(roots):
    return _compute_c4(roots)

def _objective_grad(roots):
    c4, g = _compute_c4(roots, need_grad=True)
    if not np.isfinite(c4):
        return np.inf, np.zeros_like(roots)
    return c4, g

def _c4_from_alpha(alpha):
    """Compute C4 for roots of generalized Laguerre polynomial L_12^{(alpha)}."""
    try:
        xg, _ = roots_genlaguerre(12, alpha)
        xg = np.sort(xg)
        if not np.all(xg > 0) or not np.isfinite(xg).all():
            return np.inf
        return _compute_c4(xg)
    except Exception:
        return np.inf

def _log_param_objective(log_roots):
    """Objective in log-space: exp(log_roots) are the actual roots."""
    roots = np.exp(log_roots)
    c4 = _compute_c4(roots)
    return c4

def _log_param_grad(log_roots):
    """Gradient in log-space: dC4/d(log xi) = xi * dC4/dxi."""
    roots = np.exp(log_roots)
    c4, g = _compute_c4(roots, need_grad=True)
    if not np.isfinite(c4):
        return np.inf, np.zeros_like(log_roots)
    # Chain rule: dC4/d(log xi) = roots[i] * dC4/dxi
    log_grad = roots * g
    return c4, log_grad

def construct():
    start = time.time()
    DEADLINE = 630.0
    SAFETY = 30.0
    deadline = start + DEADLINE - SAFETY

    def time_ok():
        return (time.time() - start) < deadline

    # Phase 1: 1D search over alpha to find optimal root density profile
    best_alpha = 0.0
    best_c4_alpha = np.inf
    
    # Coarse grid
    log_alphas = np.linspace(math.log(1e-3), math.log(1e4), 100)
    for la in log_alphas:
        if not time_ok():
            break
        alpha = np.exp(la)
        c4_val = _c4_from_alpha(alpha)
        if c4_val < best_c4_alpha:
            best_c4_alpha = c4_val
            best_alpha = alpha

    # Refine with scalar minimization
    if time_ok():
        try:
            def _log_alpha_objective(log_a):
                alpha = np.exp(log_a)
                if alpha < 1e-3 or alpha > 1e4:
                    return np.inf
                return _c4_from_alpha(alpha)

            res_alpha = minimize_scalar(
                _log_alpha_objective,
                bounds=(math.log(max(best_alpha/10, 1e-3)), math.log(min(best_alpha*10, 1e4))),
                method='bounded',
                options={'xatol': 1e-8, 'maxiter': 100}
            )
            if res_alpha.success and np.isfinite(res_alpha.fun) and res_alpha.fun < best_c4_alpha:
                best_alpha = np.exp(res_alpha.x)
                best_c4_alpha = res_alpha.fun
        except Exception:
            pass

    # Get roots from best alpha
    xg, _ = roots_genlaguerre(12, best_alpha)
    best_roots = np.sort(xg)
    best_c4 = _compute_c4(best_roots)

    # Phase 2: Full 12-DOF L-BFGS-B refinement in log-space from the alpha-optimal point
    # Log-space removes positivity constraints and can improve convergence
    if time_ok():
        try:
            log_roots_init = np.log(best_roots)
            res = minimize(
                _log_param_grad,
                log_roots_init,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            if np.isfinite(res.fun):
                candidate = np.sort(np.exp(res.x))
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 2b: Also try in original space for comparison
    if time_ok():
        try:
            res = minimize(
                _objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-8, None)] * 12,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 3: Try a few other alpha values to see if we missed a better basin
    if time_ok():
        for factor in [0.1, 0.5, 2.0, 10.0]:
            a_try = best_alpha * factor
            if a_try < 1e-3 or a_try > 1e4:
                continue
            try:
                xg2, _ = roots_genlaguerre(12, a_try)
                xg2 = np.sort(xg2)
                if not np.all(xg2 > 0):
                    continue
                # Try both log-space and original space
                for space in ['log', 'orig']:
                    if not time_ok():
                        break
                    try:
                        if space == 'log':
                            log_init = np.log(xg2)
                            res2 = minimize(
                                _log_param_grad,
                                log_init,
                                jac=True,
                                method='L-BFGS-B',
                                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 40}
                            )
                            if np.isfinite(res2.fun):
                                candidate = np.sort(np.exp(res2.x))
                        else:
                            res2 = minimize(
                                _objective_grad,
                                xg2,
                                jac=True,
                                method='L-BFGS-B',
                                bounds=[(1e-8, None)] * 12,
                                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 40}
                            )
                            if np.all(res2.x > 0) and np.isfinite(res2.fun):
                                candidate = np.sort(res2.x)
                            else:
                                continue
                        c4 = _compute_c4(candidate)
                        if c4 < best_c4:
                            best_c4 = c4
                            best_roots = candidate.copy()
                    except Exception:
                        pass
            except Exception:
                pass

    # Phase 4: Final high-precision polish in log-space
    for polish_iter in range(5):
        if not time_ok():
            break
        try:
            log_roots = np.log(best_roots)
            res = minimize(
                _log_param_grad,
                log_roots,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 50}
            )
            if np.isfinite(res.fun):
                candidate = np.sort(np.exp(res.x))
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
                else:
                    break
        except Exception:
            break

    # Phase 5: Additional polish in original space
    for polish_iter in range(3):
        if not time_ok():
            break
        try:
            res = minimize(
                _objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-10, None)] * 12,
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 50}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
                else:
                    break
        except Exception:
            break

    # Fallback
    if not np.all(np.isfinite(best_roots)) or not np.all(best_roots > 0):
        x0, _ = roots_genlaguerre(12, 0.0)
        best_roots = np.sort(x0)
    
    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
