"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_genlaguerre
from scipy.optimize import minimize, differential_evolution
import time

# Precompute Gauss-Laguerre quadrature nodes and weights (weight e^{-x})
_N_QUAD = 400
_x_quad, _w_quad = np.polynomial.laguerre.laggauss(_N_QUAD)

def _compute_ratio_and_grad(roots):
    """
    Compute the C4 ratio R = ||P||_2^2 / ||P||_1^2 for P(x)=prod(x-r_i)^2
    with weight e^{-x}, plus dR/droots. Returns (R, grad).
    """
    roots = np.asarray(roots, dtype=np.float64)
    n_roots = len(roots)

    # log P(x) = 2 * sum_j log|x - r_j|
    log_p = np.zeros(_N_QUAD)
    for r in roots:
        diff = np.abs(_x_quad - r)
        diff[diff < 1e-300] = 1e-300
        log_p += 2.0 * np.log(diff)

    log_p2 = 2.0 * log_p
    log_w = np.log(_w_quad)  # log of quadrature weight (includes e^{-x})

    # I1 = sum w_i P_i ; I2 = sum w_i P_i^2
    def _stable_sum(log_vals):
        m = np.max(log_vals)
        if not np.isfinite(m):
            return -np.inf, -np.inf
        s = np.sum(np.exp(log_vals - m))
        if s <= 0:
            return -np.inf, -np.inf
        return m + np.log(s), np.exp(log_vals - m) / s  # value, normalized weights

    log_I1 = log_w + log_p
    log_I2 = log_w + log_p2
    I1, n1 = _stable_sum(log_I1)
    I2, n2 = _stable_sum(log_I2)
    if not (np.isfinite(I1) and np.isfinite(I2)) or I1 <= 0:
        return np.inf, np.zeros(n_roots)

    R = np.exp(I2 - 2.0 * I1)

    # d log I1 / d r_j = sum_i n1_i * 2/(x_i - r_j)
    # d log I2 / d r_j = sum_i n2_i * 4/(x_i - r_j)
    # dR/dr_j = R * (dlogI2 - 2 dlogI1)
    grad = np.zeros(n_roots)
    for j, r in enumerate(roots):
        diff = _x_quad - r
        diff[diff < 1e-300] = 1e-300
        diff[diff > -1e-300] = -1e-300
        inv = 1.0 / diff
        dlogI1 = 2.0 * np.sum(n1 * inv)
        dlogI2 = 4.0 * np.sum(n2 * inv)
        grad[j] = R * (dlogI2 - 2.0 * dlogI1)
    return R, grad

def _compute_ratio(roots):
    v, _ = _compute_ratio_and_grad(roots)
    return v

def _roots_genlag(beta):
    """Nodes of the generalized Laguerre rule for weight x^beta e^{-x}."""
    try:
        x, _ = roots_genlaguerre(12, beta)
        return np.sort(x)
    except Exception:
        return None

def _roots_logaffine(a, b):
    """12 roots affine in log-space: r_i = exp(a + b*i)."""
    i = np.arange(12)
    return np.exp(a + b * i)

def construct():
    start_time = time.time()
    deadline = start_time + 585.0

    best_roots = None
    best_R = np.inf

    def _update(roots):
        nonlocal best_roots, best_R
        roots = np.sort(np.asarray(roots, dtype=np.float64))
        if len(roots) != 12 or np.any(~np.isfinite(roots)) or np.any(roots <= 0):
            return
        R = _compute_ratio(roots)
        if np.isfinite(R) and R < best_R:
            best_R = R
            best_roots = roots.copy()

    # ---- Phase 1: 1-DOF global search over generalized-Laguerre beta ----
    def obj_beta(beta):
        if time.time() > deadline - 120:
            return 1e30
        r = _roots_genlag(beta[0])
        if r is None or np.any(r <= 0) or np.any(r > 1e6):
            return 1e30
        R = _compute_ratio(r)
        return R if np.isfinite(R) else 1e30

    beta_candidates = []
    try:
        de = differential_evolution(
            obj_beta, [(0.0, 20.0)], seed=42, maxiter=200, tol=1e-12,
            popsize=20, init='sobol', polish=False
        )
        if de.fun < 1e29:
            beta_candidates.append(de.x[0])
    except Exception:
        pass

    # Broad deterministic beta sweep to be safe
    for b in np.arange(0.0, 20.0, 0.5):
        beta_candidates.append(b)
    for b in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]:
        beta_candidates.append(b)

    for b in beta_candidates:
        if time.time() > deadline - 120:
            break
        r = _roots_genlag(b)
        if r is not None:
            _update(r)

    # ---- Phase 2: 2-DOF log-affine global search ----
    def obj_la(p):
        if time.time() > deadline - 120:
            return 1e30
        a, b = p
        r = _roots_logaffine(a, b)
        if np.any(r <= 0) or np.any(r > 1e6):
            return 1e30
        R = _compute_ratio(r)
        return R if np.isfinite(R) else 1e30

    try:
        de2 = differential_evolution(
            obj_la, [(-1.0, 5.0), (-0.5, 1.0)], seed=7, maxiter=200,
            tol=1e-12, popsize=18, init='sobol', polish=False
        )
        if de2.fun < 1e29:
            _update(_roots_logaffine(de2.x[0], de2.x[1]))
    except Exception:
        pass

    # ---- Phase 3: full 12-DOF log-root refinement (removes positivity constraint) ----
    def obj_log(p):
        if time.time() > deadline - 40:
            return 1e30
        r = np.exp(p)
        if np.any(r > 1e6):
            return 1e30
        R = _compute_ratio(r)
        return R if np.isfinite(R) else 1e30

    def grad_log(p):
        if time.time() > deadline - 40:
            return np.zeros(12)
        r = np.exp(p)
        if np.any(r > 1e6):
            return np.zeros(12)
        R, g = _compute_ratio_and_grad(r)
        # dR/dp_j = dR/dr_j * r_j
        return g * r

    # Build start points from best so far plus diverse parametric seeds
    starts = []
    if best_roots is not None:
        starts.append(np.log(best_roots))
    for b in [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]:
        r = _roots_genlag(b)
        if r is not None:
            starts.append(np.log(r))
    for a in [0.0, 1.0, 2.0, 3.0]:
        for bb in [0.1, 0.2, 0.3, 0.5]:
            r = _roots_logaffine(a, bb)
            if np.all(r > 0) and np.all(r < 1e6):
                starts.append(np.log(r))

    for sp in starts:
        if time.time() > deadline - 60:
            break
        try:
            res = minimize(
                obj_log, sp, jac=grad_log, method='L-BFGS-B',
                bounds=[(-30.0, 13.0)] * 12,
                options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-16, 'maxls': 80}
            )
            if res.fun < 1e29:
                _update(np.exp(res.x))
        except Exception:
            pass

    # ---- Phase 4: high-precision final polish ----
    if best_roots is not None and time.time() < deadline - 15:
        try:
            res = minimize(
                obj_log, np.log(best_roots), jac=grad_log, method='L-BFGS-B',
                bounds=[(-30.0, 13.0)] * 12,
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-18, 'maxls': 200}
            )
            if res.fun < 1e29:
                _update(np.exp(res.x))
        except Exception:
            pass

    # Fallback
    if best_roots is None or len(best_roots) != 12 or np.any(best_roots <= 0):
        x0, _ = roots_genlaguerre(12, 0.0)
        best_roots = np.sort(x0)

    best_roots = np.sort(best_roots)
    return best_roots.tolist()
# EVOLVE-BLOCK-END
