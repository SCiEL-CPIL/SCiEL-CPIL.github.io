"""
Refined optimization for C4 upper bound with 12 double Laguerre roots.
Combines linear and log-space differential evolution with high-precision 
L-BFGS-B polishing using analytical gradients.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_laguerre, roots_genlaguerre

# Precompute factorials for moment computation up to degree 48 (P^2 is degree 48)
_MAX_DEG = 48
_FACTS = np.array([math.factorial(k) for k in range(_MAX_DEG + 1)], dtype=np.float64)

def _poly_moment(coeffs):
    """
    Compute integral of poly (ascending coeff) * e^{-x} dx.
    int (sum c_k x^k) e^{-x} dx = sum c_k k!
    """
    n = min(len(coeffs), len(_FACTS))
    if n == 0:
        return 0.0
    return float(np.dot(coeffs[:n], _FACTS[:n]))

def _build_p_coeffs(roots):
    """
    Build P(x) = prod (x-x_i)^2 in ascending coeff order.
    Returns P_coeffs (degree 24).
    """
    from numpy.polynomial.polynomial import polyfromroots, polymul
    B = polyfromroots(roots)  # degree 12, ascending
    P = polymul(B, B)          # degree 24
    return P

def _compute_c4(roots, need_grad=False):
    """
    Compute C4 = I2/I1^2 where I1=int P e^{-x}, I2=int P^2 e^{-x}, P=prod(x-x_i)^2.
    """
    roots = np.asarray(roots, dtype=np.float64)
    n = len(roots)
    
    # Basic validity check
    if np.any(roots <= 0) or np.any(~np.isfinite(roots)):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    # Sort roots to ensure consistency, though P is symmetric
    # Sorting is crucial for gradient index mapping if we were using ordered roots,
    # but here roots are just a set. However, for stability in DE, we might want to keep them sorted.
    # The objective is permutation invariant.
    
    P = _build_p_coeffs(roots)
    I1 = _poly_moment(P)
    if I1 <= 0 or not np.isfinite(I1):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    # P^2 = P * P
    from numpy.polynomial.polynomial import polymul
    P2 = polymul(P, P)
    I2 = _poly_moment(P2)
    if not np.isfinite(I2):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    c4 = I2 / (I1 * I1)
    if not need_grad:
        return float(c4)

    # Gradient computation
    # dC4/dx_i = (dI2*I1^2 - I2*2*I1*dI1)/I1^4 = dI2/I1^2 - 2*I2*dI1/I1^3
    
    grad = np.zeros(n)
    inv_I1 = 1.0 / I1
    inv_I1_sq = inv_I1 * inv_I1
    inv_I1_cu = inv_I1_sq * inv_I1

    # Precompute P coefficients for reuse
    # P is degree 24. S_i = P / (x - r_i) is degree 23.
    # T_i = P * S_i = P^2 / (x - r_i) is degree 47.
    
    for i in range(n):
        r = roots[i]
        P_arr = P
        deg = len(P_arr) - 1
        S_i = np.zeros(deg)  # degree 23
        
        # Synthetic division of P by (x - r)
        # P is ascending: P[0] + P[1]x + ... + P[24]x^24
        # We want Q such that P = (x-r)Q + R. Since r is a root, R=0.
        # Q[23] = P[24]
        # Q[k] = P[k+1] + r*Q[k+1]
        
        # Let's do it carefully from high to low degree for Q
        # P(x) = sum_{k=0}^{24} p_k x^k
        # (x-r)Q(x) = x Q(x) - r Q(x)
        # Coeff of x^24: q_23 = p_24
        # Coeff of x^k: q_k - r q_{k-1} = p_k => q_k = p_k + r q_{k-1} ? No.
        # Standard synthetic division for descending:
        # For ascending:
        # q_{deg-1} = p_deg
        # q_k = p_{k+1} + r * q_{k+1} for k = deg-2 ... 0
        
        carry = P_arr[deg]
        for k in range(deg - 1, -1, -1):
            S_i[k] = carry
            carry = P_arr[k] + carry * r
        
        # dI1_i = -2 * int S_i e^{-x} dx
        dI1_i = -2.0 * _poly_moment(S_i)
        
        # T_i = P * S_i
        T_i = polymul(P, S_i)
        # dI2_i = -4 * int T_i e^{-x} dx
        dI2_i = -4.0 * _poly_moment(T_i)
        
        grad[i] = dI2_i * inv_I1_sq - 2.0 * I2 * dI1_i * inv_I1_cu

    return float(c4), grad

def _objective_linear(roots):
    """Objective for linear space optimization."""
    c4 = _compute_c4(roots)
    if not np.isfinite(c4):
        return 1e10
    return c4

def _objective_log(log_roots):
    """Objective for log-space optimization."""
    roots = np.exp(log_roots)
    c4 = _compute_c4(roots)
    if not np.isfinite(c4):
        return 1e10
    return c4

def _objective_grad_linear(roots):
    """Objective and gradient for linear space."""
    c4, g = _compute_c4(roots, need_grad=True)
    if not np.isfinite(c4):
        return 1e10, np.zeros_like(roots)
    return c4, g

def _objective_grad_log(log_roots):
    """Objective and gradient for log space."""
    roots = np.exp(log_roots)
    c4, g_roots = _compute_c4(roots, need_grad=True)
    if not np.isfinite(c4):
        return 1e10, np.zeros_like(log_roots)
    # Chain rule: dC4/dlog_r_i = dC4/dr_i * dr_i/dlog_r_i = g_i * r_i
    g_log = g_roots * roots
    return c4, g_log

def construct():
    start = time.time()
    DEADLINE = 600.0
    SAFETY = 30.0
    deadline = start + DEADLINE - SAFETY

    def time_ok():
        return (time.time() - start) < deadline

    n_roots = 12
    best_roots = None
    best_c4 = np.inf

    # Phase 1: Initial guesses
    # 1. Standard Laguerre roots
    x0, _ = roots_laguerre(n_roots)
    x0 = np.sort(x0)
    c4_0 = _compute_c4(x0)
    if c4_0 < best_c4:
        best_c4 = c4_0
        best_roots = x0.copy()

    # 2. Generalized Laguerre roots with various alpha
    for alpha in [0.5, 1.0, 2.0, 5.0]:
        try:
            xg, _ = roots_genlaguerre(n_roots, alpha)
            xg = np.sort(xg)
            if np.all(xg > 0):
                c4g = _compute_c4(xg)
                if c4g < best_c4:
                    best_c4 = c4g
                    best_roots = xg.copy()
        except Exception:
            pass

    # Phase 2: Global Search
    # Strategy A: Linear space DE
    if time_ok():
        bounds_lin = [(1e-3, 100.0)] * n_roots
        try:
            de_res_lin = differential_evolution(
                _objective_linear,
                bounds=bounds_lin,
                seed=42,
                maxiter=150,
                popsize=15,
                tol=1e-8,
                polish=False,
                mutation=(0.5, 1.0),
                recombination=0.7
            )
            if np.all(de_res_lin.x > 0) and np.isfinite(de_res_lin.fun):
                cand = np.sort(de_res_lin.x)
                c4_cand = _compute_c4(cand)
                if c4_cand < best_c4:
                    best_c4 = c4_cand
                    best_roots = cand.copy()
        except Exception:
            pass

    # Strategy B: Log-space DE (often better for scale-invariant problems)
    if time_ok():
        # Log bounds: exp(-3) ~ 0.05, exp(5) ~ 148
        bounds_log = [(-3.0, 5.0)] * n_roots
        try:
            de_res_log = differential_evolution(
                _objective_log,
                bounds=bounds_log,
                seed=137,
                maxiter=150,
                popsize=15,
                tol=1e-8,
                polish=False,
                mutation=(0.5, 1.0),
                recombination=0.7
            )
            if np.isfinite(de_res_log.fun):
                cand_log = de_res_log.x
                cand = np.sort(np.exp(cand_log))
                c4_cand = _compute_c4(cand)
                if c4_cand < best_c4:
                    best_c4 = c4_cand
                    best_roots = cand.copy()
        except Exception:
            pass

    # Phase 3: Local Polish
    # We will polish the best root found so far, and also a few diverse candidates.
    
    # Generate candidates for polishing
    candidates = [best_roots.copy()]
    
    # Add perturbed versions of the best
    rng = np.random.default_rng(42)
    for scale in [0.1, 0.3]:
        for _ in range(2):
            pert = best_roots * (1.0 + scale * rng.standard_normal(n_roots))
            pert = np.maximum(pert, 1e-3)
            pert = np.sort(pert)
            candidates.append(pert)
            
    # Add best Laguerre roots if not already there
    if best_c4 > c4_0:
        candidates.append(x0.copy())

    # Polish each candidate with L-BFGS-B in linear space
    for init in candidates:
        if not time_ok():
            break
        try:
            res = minimize(
                _objective_grad_linear,
                init,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-6, None)] * n_roots,
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                cand = np.sort(res.x)
                c4_cand = _compute_c4(cand)
                if c4_cand < best_c4:
                    best_c4 = c4_cand
                    best_roots = cand.copy()
        except Exception:
            pass

    # Polish in log-space as well, starting from log(best_roots)
    if time_ok():
        try:
            log_init = np.log(best_roots)
            res_log = minimize(
                _objective_grad_log,
                log_init,
                jac=True,
                method='L-BFGS-B',
                bounds=[(-10.0, 10.0)] * n_roots, # Wide bounds in log space
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            if np.isfinite(res_log.fun):
                cand = np.sort(np.exp(res_log.x))
                c4_cand = _compute_c4(cand)
                if c4_cand < best_c4:
                    best_c4 = c4_cand
                    best_roots = cand.copy()
        except Exception:
            pass

    # Phase 4: Final High-Precision Polish
    # Repeat L-BFGS-B with tighter tolerances on the very best result
    for _ in range(3):
        if not time_ok():
            break
        try:
            res_final = minimize(
                _objective_grad_linear,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-8, None)] * n_roots,
                options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-15, 'maxls': 50}
            )
            if np.all(res_final.x > 0) and np.isfinite(res_final.fun):
                cand = np.sort(res_final.x)
                c4_cand = _compute_c4(cand)
                if c4_cand < best_c4:
                    best_c4 = c4_cand
                    best_roots = cand.copy()
                else:
                    break
        except Exception:
            break

    # Safety check
    if not np.all(np.isfinite(best_roots)) or not np.all(best_roots > 0):
        x0, _ = roots_laguerre(n_roots)
        best_roots = np.sort(x0)

    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
