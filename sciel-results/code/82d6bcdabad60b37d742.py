"""Analytic uncertainty bound with exact gradient + L-BFGS-B + differential_evolution."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 200

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

_signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

def _compute_p(params):
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
    return p

def _moments(p):
    I = _I
    m0 = 0.0
    m1 = 0.0
    m2 = 0.0
    for i in range(7):
        pi = p[i]
        if pi == 0.0:
            continue
        for j in range(7):
            pp = pi * p[j]
            m0 += pp * I[i+j]
            m1 += pp * I[i+j+1]
            m2 += pp * I[i+j+2]
    return m0, m1, m2

def _freq_moments(q):
    I = _I
    n0 = 0.0
    n1 = 0.0
    n2 = 0.0
    for i in range(7):
        qi = q[i]
        if qi == 0.0:
            continue
        for j in range(7):
            qq = qi * q[j]
            n0 += qq * I[i+j]
            n1 += qq * I[i+j+1]
            n2 += qq * I[i+j+2]
    n0 *= 2.0 * np.pi
    n1 *= 2.0 * np.pi
    n2 *= 2.0 * np.pi
    return n0, n1, n2

def _bound_and_grad(params):
    p = _compute_p(params)
    m0, m1, m2 = _moments(p)
    q = p * _signs
    n0, n1, n2 = _freq_moments(q)
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m1 / m0
    vF = n1 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    I = _I
    grad = np.zeros(3)
    for c in range(3):
        dp = _dp_da[:, c]
        
        dm0 = 0.0
        dm1 = 0.0
        dm2 = 0.0
        for i in range(7):
            dpi = dp[i]
            if dpi == 0.0:
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
            if dqi == 0.0:
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

def _lbfgs_from(start, maxiter=5000, ftol=1e-15, gtol=1e-14):
    try:
        res = minimize(_objective_grad, start, method='L-BFGS-B',
                      jac=True, options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
        if res.fun < np.inf and np.isfinite(res.fun):
            return res.x, res.fun
    except Exception:
        pass
    return None, np.inf

def construct():
    best_val = np.inf
    best_params = None
    
    def _try_update(params, val):
        nonlocal best_val, best_params
        if val < best_val and np.isfinite(val):
            best_val = val
            best_params = params.copy()
    
    # Phase 1: Differential Evolution for global exploration
    bounds = [(-5.0, 5.0), (-0.5, 0.5), (-0.01, 0.01)]
    if time.time() < _DEADLINE - 60:
        try:
            def _de_callback(xk, convergence):
                val, _ = _bound_and_grad(xk)
                _try_update(xk, val)
                return time.time() > _DEADLINE - 30
            
            de_res = differential_evolution(_objective, bounds, maxiter=100,
                                            tol=1e-10, popsize=15, seed=42,
                                            callback=_de_callback,
                                            init='sobol')
            if de_res.fun < best_val and np.isfinite(de_res.fun):
                best_val = de_res.fun
                best_params = de_res.x.copy()
        except Exception:
            pass
    
    # Phase 2: L-BFGS-B from multiple starts
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
    ]
    
    rng = np.random.default_rng(123)
    for i in range(15):
        scale = 0.2 + 0.4 * i / 15
        starts.append(rng.normal(0, scale, 3))
    
    if best_params is not None:
        starts.append(best_params)
    
    for start in starts:
        if time.time() > _DEADLINE - 25:
            break
        params, val = _lbfgs_from(start, maxiter=3000, ftol=1e-14, gtol=1e-13)
        if params is not None:
            _try_update(params, val)
    
    # Phase 3: Polish best with tighter tolerances
    if best_params is not None and time.time() < _DEADLINE - 15:
        params, val = _lbfgs_from(best_params, maxiter=10000, ftol=1e-16, gtol=1e-15)
        if params is not None:
            _try_update(params, val)
    
    # Phase 4: Nelder-Mead polish
    if best_params is not None and time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Final L-BFGS-B polish
    if best_params is not None and time.time() < _DEADLINE - 5:
        params, val = _lbfgs_from(best_params, maxiter=10000, ftol=1e-16, gtol=1e-15)
        if params is not None:
            _try_update(params, val)
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
