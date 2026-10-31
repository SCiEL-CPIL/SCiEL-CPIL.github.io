"""Uncertainty inequality optimization: structure-aware search in a reparameterized space.

Key structural change: instead of optimizing (a0, a4, a8) directly, we optimize a
basis that respects the constraint a12 = -(a0+3a4+105a8)/945 more naturally, and
we use a global search (differential evolution) combined with intensive local
polishing. The reparameterization makes the landscape smoother and more amenable
to global methods.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 195  # 15s safety margin before 210s kill


def _precompute_I():
    """Precompute I[n] = integral x^(2n) e^(-x^2) dx for n=0..12."""
    I = np.zeros(13)
    I[0] = np.sqrt(np.pi)
    for n in range(1, 13):
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
    p = np.zeros(7)
    p[0] = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
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
    """
    m = np.zeros(3)
    n = np.zeros(3)
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


def _bound_and_grad(params):
    """Compute bound and gradient via central finite differences."""
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


def _local_optimize(x0, maxiter=5000):
    """Run L-BFGS-B local optimization from x0."""
    try:
        res = minimize(_objective_grad, x0, method='L-BFGS-B',
                       jac=True, options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-15})
        if res.fun < 1e17 and np.isfinite(res.fun):
            return res.x, res.fun
    except Exception:
        pass
    return x0, _bound_from_params(x0)


def construct():
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642])
    
    rng = np.random.default_rng(42)
    t0 = time.time()
    
    # Phase 1: Differential evolution for global exploration
    # The parameter space is 3D; we define reasonable bounds
    bounds = [(-4.0, 3.0), (-0.5, 0.5), (-0.05, 0.05)]
    
    # Use differential evolution with a moderate population
    if time.time() - t0 < _DEADLINE - 60:
        try:
            de_res = differential_evolution(_objective, bounds,
                                            maxiter=200, popsize=30,
                                            tol=1e-12, mut=(0.3, 1.2),
                                            seed=42, polish=True,
                                            init='sobol')
            if de_res.fun < best_val and np.isfinite(de_res.fun):
                best_val = de_res.fun
                best_params = de_res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Basin hopping around the best point found
    current_best = best_params.copy()
    current_val = best_val
    
    scales = [0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.03, 0.02, 0.01, 0.005]
    
    for scale in scales:
        if time.time() - t0 > _DEADLINE - 30:
            break
        
        n_trials = max(3, int(10 * scale / 0.02))
        n_trials = min(n_trials, 15)
        
        for _ in range(n_trials):
            if time.time() - t0 > _DEADLINE - 30:
                break
            
            x_pert = current_best + rng.normal(0, scale, 3)
            x_opt, f_opt = _local_optimize(x_pert)
            
            if f_opt < current_val:
                current_val = f_opt
                current_best = x_opt.copy()
                
                if f_opt < best_val:
                    best_val = f_opt
                    best_params = x_opt.copy()
    
    # Phase 3: Intensive local polishing from multiple nearby points
    if time.time() - t0 < _DEADLINE - 15:
        polish_starts = [best_params.copy()]
        for scale in [0.01, 0.005, 0.001]:
            for _ in range(5):
                polish_starts.append(best_params + rng.normal(0, scale, 3))
        
        for ps in polish_starts:
            if time.time() - t0 > _DEADLINE - 15:
                break
            x_opt, f_opt = _local_optimize(ps, maxiter=10000)
            if f_opt < best_val:
                best_val = f_opt
                best_params = x_opt.copy()
    
    # Phase 4: Nelder-Mead for a different local optimizer perspective
    if time.time() - t0 < _DEADLINE - 5:
        try:
            nm_res = minimize(_objective, best_params, method='Nelder-Mead',
                            options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18, 'adaptive': True})
            if nm_res.fun < best_val and np.isfinite(nm_res.fun):
                best_val = nm_res.fun
                best_params = nm_res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Final ultra-precise L-BFGS-B polishing
    if time.time() - t0 < _DEADLINE:
        try:
            fin_res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                             jac=True, options={'maxiter': 50000, 'ftol': 1e-18, 'gtol': 1e-17})
            if fin_res.fun < best_val and np.isfinite(fin_res.fun):
                best_val = fin_res.fun
                best_params = fin_res.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
