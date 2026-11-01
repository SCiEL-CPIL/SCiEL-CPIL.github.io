"""Analytic uncertainty bound with exact gradient + L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    
    A(f)^2 = <x^2>/<1> for |g|^2
    A(F)^2 = <xi^2>/<1> for |G|^2
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # Coefficients of P(x) = sum_{k=0}^{6} p_k x^{2k}
    p0 = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    p1 = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p2 = a4 + 210.0*a8 + 13860.0*a12
    p3 = -28.0*a8 - 6930.0*a12
    p4 = a8 + 1155.0*a12
    p5 = -66.0*a12
    p6 = a12
    p = np.array([p0, p1, p2, p3, p4, p5, p6])
    
    # Precompute I_n = integral x^{2n} e^{-x^2} dx
    sp = np.sqrt(np.pi)
    I = np.zeros(14)
    # I[n] = (2n-1)!! * sqrt(pi) / 2^n for n >= 1, I[0] = sqrt(pi)
    # More precisely: I[n] = (2n)! / (4^n * n!) * sqrt(pi)
    # Using recurrence: I[0] = sqrt(pi), I[n] = (2n-1)/(2n) * I[n-1]
    I[0] = sp
    for n in range(1, 14):
        I[n] = (2*n - 1) / (2*n) * I[n-1]
    
    # m_k = sum_{i,j=0}^{6} p_i p_j I[k+i+j]
    m = np.zeros(3)
    for i in range(7):
        for j in range(7):
            pp = p[i] * p[j]
            for k in range(3):
                idx = k + i + j
                if idx < 14:
                    m[k] += pp * I[idx]
    
    # For Fourier transform: Q(xi) = sum q_k xi^{2k} where q_k = p_k * (-1)^k
    q = np.zeros(7)
    for k in range(7):
        q[k] = p[k] * ((-1.0)**k)
    
    # |G|^2 = 2*pi * |Q|^2 * e^{-xi^2}
    # n_k = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    n = np.zeros(3)
    for i in range(7):
        for j in range(7):
            qq = q[i] * q[j]
            for k in range(3):
                idx = k + i + j
                if idx < 14:
                    n[k] += 2.0*np.pi * qq * I[idx]
    
    m0, m2, m4 = m[0], m[1], m[2]
    n0, n2, n4 = n[0], n[1], n[2]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Compute gradient analytically
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    grad = np.zeros(3)
    for c in range(3):
        dp = np.zeros(7)
        dp[0] = np.array([1.0, 3.0, 105.0])[c] + 945.0*da12[c]
        dp[1] = np.array([0.0, -6.0, -420.0])[c] - 7560.0*da12[c]
        dp[2] = np.array([0.0, 1.0, 210.0])[c] + 13860.0*da12[c]
        dp[3] = np.array([0.0, 0.0, -28.0])[c] - 6930.0*da12[c]
        dp[4] = np.array([0.0, 0.0, 1.0])[c] + 1155.0*da12[c]
        dp[5] = np.array([0.0, 0.0, 0.0])[c] - 66.0*da12[c]
        dp[6] = np.array([0.0, 0.0, 0.0])[c] + 1.0*da12[c]
        
        dm = np.zeros(3)
        for i in range(7):
            dpi = dp[i]
            if dpi == 0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 14:
                        s += p[j] * I[idx]
                dm[k] += 2.0 * dpi * s
        
        dq = np.zeros(7)
        for k in range(7):
            dq[k] = dp[k] * ((-1.0)**k)
        
        dn = np.zeros(3)
        for i in range(7):
            dqi = dq[i]
            if dqi == 0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 14:
                        s += q[j] * I[idx]
                dn[k] += 2.0 * 2.0*np.pi * dqi * s
        
        dvm2 = (dm[1]*m0 - m2*dm[0]) / (m0*m0)
        dvn2 = (dn[1]*n0 - n2*dn[0]) / (n0*n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def construct():
    # Multiple starting points including the seed and diverse perturbations
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-1.13, 0.04, 0.000308642] + 0.1*np.array([1, 0.5, 0.01])),
        np.array([-1.13, 0.04, 0.000308642] - 0.1*np.array([1, 0.5, 0.01])),
        np.array([-1.13, 0.04, 0.000308642] + 0.5*np.array([0.1, 1, 0.001])),
        np.array([-1.13, 0.04, 0.000308642] - 0.5*np.array([0.1, 1, 0.001])),
    ]
    
    rng = np.random.default_rng(123)
    for i in range(30):
        starts.append(rng.normal(0, 0.8, 3))
    
    best_val = np.inf
    best_params = None
    
    for start in starts:
        if time.time() > _DEADLINE - 25:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Polish with Nelder-Mead from best
    if best_params is not None and time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final polish with L-BFGS-B at tighter tolerance
    if best_params is not None and time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Second round of local search from best with small perturbations
    if best_params is not None and time.time() < _DEADLINE - 5:
        for i in range(10):
            if time.time() > _DEADLINE - 5:
                break
            perturbation = rng.normal(0, 0.01, 3)
            trial_start = best_params + perturbation
            try:
                res = minimize(_objective_grad, trial_start, method='L-BFGS-B',
                              jac=True, options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-14})
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
