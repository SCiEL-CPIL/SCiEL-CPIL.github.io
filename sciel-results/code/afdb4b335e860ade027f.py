"""Uncertainty inequality optimization with log-magnitude parameterization and multi-phase search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 195  # 15s safety margin before 210s kill


def _precompute_I():
    """Precompute I[n] = integral x^(2n) e^(-x^2) dx for n=0..12."""
    I = np.zeros(13)
    for n in range(13):
        if n == 0:
            I[0] = np.sqrt(np.pi)
        else:
            I[n] = (2.0 * n - 1.0) / 2.0 * I[n - 1]
    return I


_I = _precompute_I()


def _coeffs_from_params(a0, a4, a8):
    """
    Compute polynomial coefficients p_k of P(x) = sum p_k x^(2k)
    where P = a0*H0 + a4*H4 + a8*H8 + a12*H12
    and a12 = -(a0 + 3*a4 + 105*a8)/945
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    p = np.zeros(7)
    p[0] = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12  # should be ~0
    p[1] = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p[2] = a4 + 210.0 * a8 + 13860.0 * a12
    p[3] = -28.0 * a8 - 6930.0 * a12
    p[4] = a8 + 1155.0 * a12
    p[5] = -66.0 * a12
    p[6] = a12
    return p


def _compute_bound(p):
    """
    Compute the uncertainty product sqrt(vf * vF) given polynomial coeffs p.
    vf = m2/m0 where m_k = integral x^k |P|^2 e^(-x^2) dx
    vF = n2/n0 where n_k = 2*pi * integral x^k |Q|^2 e^(-x^2) dx, Q_k = (-1)^k p_k
    """
    # m_k = sum_{i,j} p_i p_j I[k+i+j] for k=0,1,2
    m = np.zeros(3)
    n = np.zeros(3)
    # Q_k = (-1)^k p_k
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    
    for i in range(7):
        pi = p[i]
        qi = q[i]
        if pi == 0.0 and qi == 0.0:
            continue
        for j in range(7):
            pij = pi * p[j]
            qij = qi * q[j]
            if pij == 0.0 and qij == 0.0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 13:
                    m[k] += pij * _I[idx]
                    n[k] += qij * _I[idx]
    n *= 2.0 * np.pi
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
    if m2 <= 0 or n2 <= 0:
        return 1e18
    
    vf = m2 / m0
    vF = n2 / n0
    bound = np.sqrt(vf * vF)
    
    if not np.isfinite(bound):
        return 1e18
    return bound


def _bound_from_params(params):
    """Compute bound from (a0, a4, a8)."""
    p = _coeffs_from_params(params[0], params[1], params[2])
    return _compute_bound(p)


def _bound_from_log_params(log_params):
    """
    Compute bound from log-magnitude parameterization.
    log_params = [log_mag0, log_mag4, log_mag8, sign0, sign4, sign8]
    where a_i = sign_i * exp(log_mag_i)
    """
    a0 = log_params[3] * np.exp(log_params[0])
    a4 = log_params[4] * np.exp(log_params[1])
    a8 = log_params[5] * np.exp(log_params[2])
    return _bound_from_params(np.array([a0, a4, a8]))


def _bound_and_grad(params):
    """Compute bound and analytic gradient via finite differences (smooth problem)."""
    a0, a4, a8 = params
    f0 = _bound_from_params(params)
    h = 1e-8
    grad = np.zeros(3)
    for c in range(3):
        dp = params.copy()
        dp[c] += h
        fp = _bound_from_params(dp)
        dm = params.copy()
        dm[c] -= h
        fm = _bound_from_params(dm)
        grad[c] = (fp - fm) / (2.0 * h)
    return f0, grad


def _objective(params):
    return _bound_from_params(params)


def _objective_grad(params):
    return _bound_and_grad(params)


def _log_objective(log_params):
    """Objective in log-magnitude space."""
    return _bound_from_log_params(log_params)


def _log_objective_grad(log_params):
    """Gradient in log-magnitude space via chain rule."""
    a0, a4, a8 = _log_params_to_params(log_params)
    f0, grad_a = _bound_and_grad(np.array([a0, a4, a8]))
    # da_i/dlog_mag_i = sign_i * exp(log_mag_i) = a_i
    # da_i/dsign_i = exp(log_mag_i)
    grad_log = np.zeros(6)
    grad_log[0] = grad_a[0] * a0 / max(abs(a0), 1e-15)  # d/dlog_mag0
    grad_log[1] = grad_a[1] * a4 / max(abs(a4), 1e-15)
    grad_log[2] = grad_a[2] * a8 / max(abs(a8), 1e-15)
    grad_log[3] = grad_a[0] * np.exp(log_params[0])  # d/dsign0
    grad_log[4] = grad_a[1] * np.exp(log_params[1])
    grad_log[5] = grad_a[2] * np.exp(log_params[2])
    return f0, grad_log


def _log_params_to_params(log_params):
    """Convert log-magnitude params to linear params."""
    a0 = log_params[3] * np.exp(log_params[0])
    a4 = log_params[4] * np.exp(log_params[1])
    a8 = log_params[5] * np.exp(log_params[2])
    return a0, a4, a8


def _params_to_log_params(params):
    """Convert linear params to log-magnitude params."""
    a0, a4, a8 = params
    log_params = np.zeros(6)
    log_params[0] = np.log(max(abs(a0), 1e-15))
    log_params[1] = np.log(max(abs(a4), 1e-15))
    log_params[2] = np.log(max(abs(a8), 1e-15))
    log_params[3] = 1.0 if a0 >= 0 else -1.0
    log_params[4] = 1.0 if a4 >= 0 else -1.0
    log_params[5] = 1.0 if a8 >= 0 else -1.0
    return log_params


def construct():
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642])
    
    # Phase 1: Broad random sampling in linear space
    rng = np.random.default_rng(42)
    
    starts = []
    # Around the seed
    for _ in range(15):
        s = np.array([-1.13, 0.04, 0.000308642]) + rng.normal(0, 0.1, 3)
        starts.append(s)
    # Wider exploration
    for _ in range(30):
        s = np.array([rng.uniform(-3, 2), rng.uniform(-0.5, 0.5), rng.uniform(-0.01, 0.01)])
        starts.append(s)
    # Structured grid-like
    for a0 in [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5]:
        for a4 in [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3]:
            a8 = 0.0005 * rng.normal()
            starts.append(np.array([a0, a4, a8]))
    
    # Evaluate all starts
    for s in starts:
        if time.time() > _DEADLINE - 40:
            break
        try:
            val = _bound_from_params(s)
            if val < best_val:
                best_val = val
                best_params = s.copy()
        except Exception:
            pass
    
    # Phase 2: L-BFGS-B from best starts in linear space
    scored_starts = []
    for s in starts:
        try:
            val = _bound_from_params(s)
            scored_starts.append((val, s))
        except Exception:
            pass
    scored_starts.sort(key=lambda x: x[0])
    
    top_starts = [s for _, s in scored_starts[:10]]
    top_starts.append(best_params.copy())
    
    for s in top_starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective_grad, s, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Log-magnitude space search
    if time.time() < _DEADLINE - 35:
        # Convert best to log space
        log_best = _params_to_log_params(best_params)
        
        # Random starts in log space
        log_starts = []
        for _ in range(20):
            ls = np.array([
                rng.uniform(-8, 1),   # log_mag0
                rng.uniform(-8, 1),   # log_mag4
                rng.uniform(-12, -2), # log_mag8 (smaller magnitude)
                rng.choice([-1, 1]),
                rng.choice([-1, 1]),
                rng.choice([-1, 1])
            ])
            log_starts.append(ls)
        
        # Also add neighborhood of best in log space
        for _ in range(10):
            ls = log_best + rng.normal(0, 0.5, 6)
            ls[3] = 1.0 if ls[3] >= 0 else -1.0
            ls[4] = 1.0 if ls[4] >= 0 else -1.0
            ls[5] = 1.0 if ls[5] >= 0 else -1.0
            log_starts.append(ls)
        
        # Evaluate log starts
        log_scored = []
        for ls in log_starts:
            try:
                val = _log_objective(ls)
                log_scored.append((val, ls))
            except Exception:
                pass
        log_scored.sort(key=lambda x: x[0])
        
        # L-BFGS-B in log space from top starts
        for _, ls in log_scored[:8]:
            if time.time() > _DEADLINE - 25:
                break
            try:
                res = minimize(_log_objective_grad, ls, method='L-BFGS-B',
                              jac=True, options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-14})
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    a0, a4, a8 = _log_params_to_params(res.x)
                    best_params = np.array([a0, a4, a8])
            except Exception:
                pass
    
    # Phase 4: Differential evolution in linear space for global search
    if time.time() < _DEADLINE - 25:
        try:
            bounds = [(-4.0, 4.0), (-1.5, 1.5), (-0.02, 0.02)]
            de_res = differential_evolution(_objective, bounds, maxiter=150,
                                           popsize=20, tol=1e-12, seed=42,
                                           polish=False, init='sobol')
            if de_res.fun < best_val and np.isfinite(de_res.fun):
                best_val = de_res.fun
                best_params = de_res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Nelder-Mead polish
    if time.time() < _DEADLINE - 10:
        try:
            nm_res = minimize(_objective, best_params, method='Nelder-Mead',
                            options={'maxiter': 100000, 'xatol': 1e-16, 'fatol': 1e-18})
            if nm_res.fun < best_val and np.isfinite(nm_res.fun):
                best_val = nm_res.fun
                best_params = nm_res.x.copy()
        except Exception:
            pass
    
    # Phase 6: Final L-BFGS-B polish
    if time.time() < _DEADLINE - 5:
        try:
            fin_res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                             jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15})
            if fin_res.fun < best_val and np.isfinite(fin_res.fun):
                best_val = fin_res.fun
                best_params = fin_res.x.copy()
        except Exception:
            pass
    
    # Phase 7: One more round of local refinement with smaller steps
    if time.time() < _DEADLINE - 2:
        try:
            # Try perturbing best and re-optimizing
            for _ in range(5):
                perturbed = best_params + rng.normal(0, 1e-3, 3)
                res = minimize(_objective_grad, perturbed, method='L-BFGS-B',
                             jac=True, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15})
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best_params = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
