"""Analytic uncertainty bound with exact gradient + L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 205

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
def _precompute_I():
    I = np.zeros(14)
    sp = np.sqrt(np.pi)
    for n in range(14):
        log_val = 0.5 * np.log(np.pi)
        for m in range(1, 2*n+1):
            log_val += np.log(m)
        log_val -= n * np.log(4.0)
        for m in range(1, n+1):
            log_val -= np.log(m)
        I[n] = np.exp(log_val)
    return I

_I = _precompute_I()

# Precompute the derivative coefficient matrix
# dp_k/da_c for c=0,1,2 (a0, a4, a8)
_da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
_dp_da = np.zeros((7, 3))
_dp_da[0, 0] = 1.0 + 945.0 * _da12[0]
_dp_da[0, 1] = 3.0 + 945.0 * _da12[1]
_dp_da[0, 2] = 105.0 + 945.0 * _da12[2]
_dp_da[1, 0] = 0.0 - 7560.0 * _da12[0]
_dp_da[1, 1] = -6.0 - 7560.0 * _da12[1]
_dp_da[1, 2] = -420.0 - 7560.0 * _da12[2]
_dp_da[2, 0] = 0.0 + 13860.0 * _da12[0]
_dp_da[2, 1] = 1.0 + 13860.0 * _da12[1]
_dp_da[2, 2] = 210.0 + 13860.0 * _da12[2]
_dp_da[3, 0] = 0.0 - 6930.0 * _da12[0]
_dp_da[3, 1] = 0.0 - 6930.0 * _da12[1]
_dp_da[3, 2] = -28.0 - 6930.0 * _da12[2]
_dp_da[4, 0] = 0.0 + 1155.0 * _da12[0]
_dp_da[4, 1] = 0.0 + 1155.0 * _da12[1]
_dp_da[4, 2] = 1.0 + 1155.0 * _da12[2]
_dp_da[5, 0] = 0.0 - 66.0 * _da12[0]
_dp_da[5, 1] = 0.0 - 66.0 * _da12[1]
_dp_da[5, 2] = 0.0 - 66.0 * _da12[2]
_dp_da[6, 0] = 0.0 + 1.0 * _da12[0]
_dp_da[6, 1] = 0.0 + 1.0 * _da12[1]
_dp_da[6, 2] = 0.0 + 1.0 * _da12[2]

# Precompute (-1)^k for q_k
_signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    p = np.array([
        a0 + 3.0*a4 + 105.0*a8 + 945.0*a12,
        -6.0*a4 - 420.0*a8 - 7560.0*a12,
        a4 + 210.0*a8 + 13860.0*a12,
        -28.0*a8 - 6930.0*a12,
        a8 + 1155.0*a12,
        -66.0*a12,
        a12
    ])
    
    I = _I
    
    m0 = 0.0
    m1 = 0.0
    m2 = 0.0
    for i in range(7):
        pi = p[i]
        if pi == 0:
            continue
        for j in range(7):
            pp = pi * p[j]
            m0 += pp * I[i+j]
            m1 += pp * I[i+j+1]
            m2 += pp * I[i+j+2]
    
    q = p * _signs
    
    n0 = 0.0
    n1 = 0.0
    n2 = 0.0
    for i in range(7):
        qi = q[i]
        if qi == 0:
            continue
        for j in range(7):
            qq = qi * q[j]
            n0 += qq * I[i+j]
            n1 += qq * I[i+j+1]
            n2 += qq * I[i+j+2]
    
    n0 *= 2.0 * np.pi
    n1 *= 2.0 * np.pi
    n2 *= 2.0 * np.pi
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m1 / m0
    vF = n1 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    grad = np.zeros(3)
    for c in range(3):
        dp = _dp_da[:, c]
        
        dm0 = 0.0
        dm1 = 0.0
        dm2 = 0.0
        for i in range(7):
            dpi = dp[i]
            if dpi == 0:
                continue
            s0 = 0.0
            s1 = 0.0
            s2 = 0.0
            for j in range(7):
                pj = p[j]
                s0 += pj * I[i+j]
                s1 += pj * I[i+j+1]
                s2 += pj * I[i+j+2]
            dm0 += 2.0 * dpi * s0
            dm1 += 2.0 * dpi * s1
            dm2 += 2.0 * dpi * s2
        
        dq = dp * _signs
        dn0 = 0.0
        dn1 = 0.0
        dn2 = 0.0
        for i in range(7):
            dqi = dq[i]
            if dqi == 0:
                continue
            s0 = 0.0
            s1 = 0.0
            s2 = 0.0
            for j in range(7):
                qj = q[j]
                s0 += qj * I[i+j]
                s1 += qj * I[i+j+1]
                s2 += qj * I[i+j+2]
            dn0 += 2.0 * 2.0*np.pi * dqi * s0
            dn1 += 2.0 * 2.0*np.pi * dqi * s1
            dn2 += 2.0 * 2.0*np.pi * dqi * s2
        
        dvm2 = (dm1*m0 - m1*dm0) / (m0*m0)
        dvn2 = (dn1*n0 - n1*dn0) / (n0*n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def construct():
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-1.2, 0.03, 0.0002]),
        np.array([-0.8, 0.05, 0.0005]),
        np.array([-2.5, 0.07, 0.0007]),
        np.array([-1.8, 0.06, 0.0006]),
    ]
    
    rng = np.random.default_rng(42)
    for i in range(30):
        scale = 0.3 + 0.3 * i / 30
        starts.append(rng.normal(0, scale, 3))
    
    best_val = np.inf
    best_params = None
    
    # Phase 0: Differential evolution for global exploration
    if time.time() < _DEADLINE - 120:
        try:
            bounds = [(-5.0, 5.0), (-2.0, 2.0), (-0.01, 0.01)]
            de_result = minimize(_objective, [0.0, 0.0, 0.0], method='differential_evolution',
                                bounds=bounds, seed=42, maxiter=100, tol=1e-8,
                                popsize=15, polish=False)
            if de_result.fun < best_val and np.isfinite(de_result.fun):
                best_val = de_result.fun
                best_params = de_result.x.copy()
        except Exception:
            pass
    
    # Add DE result to starts if found
    if best_params is not None:
        starts.append(best_params.copy())
    
    # Phase 1: L-BFGS-B from all starts
    for start in starts:
        if time.time() > _DEADLINE - 40:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-13})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Polish best with tighter tolerances
    if best_params is not None and time.time() < _DEADLINE - 20:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead polish
    if best_params is not None and time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 30000, 'xatol': 1e-14, 'fatol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final L-BFGS-B polish
    if best_params is not None and time.time() < _DEADLINE - 3:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-17, 'gtol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
