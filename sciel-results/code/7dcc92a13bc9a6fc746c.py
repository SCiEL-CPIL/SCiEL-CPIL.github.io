"""Analytic uncertainty bound with precomputed constants + vectorized ops + L-BFGS-B + basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 205

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
_I_ARR = np.zeros(14)
for _n in range(14):
    if _n == 0:
        _I_ARR[_n] = np.sqrt(np.pi)
    else:
        # (2n)! / (4^n n!) * sqrt(pi)
        # Use recurrence: I_n = I_{n-1} * (2n-1)/(4n)
        _I_ARR[_n] = _I_ARR[_n-1] * (2.0*_n - 1.0) / (4.0*_n)

# Precompute the matrix M[k,i,j] = I[k+i+j] for k=0,1,2; i,j=0..6
# Shape: (3, 7, 7)
_M_MAT = np.zeros((3, 7, 7))
for _k in range(3):
    for _i in range(7):
        for _j in range(7):
            _idx = _k + _i + _j
            if _idx < 14:
                _M_MAT[_k, _i, _j] = _I_ARR[_idx]

# Precompute dp_k/da_c matrix: shape (3, 7) where rows are c=0,1,2 and cols are k=0..6
# p0 = a0+3a4+105a8+945a12
# p1 = -6a4-420a8-7560a12
# p2 = a4+210a8+13860a12
# p3 = -28a8-6930a12
# p4 = a8+1155a12
# p5 = -66a12
# p6 = a12
# da12/da0=-1/945, da12/da4=-3/945, da12/da8=-105/945
_da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
# Coefficient matrix for p in terms of (a0,a4,a8,a12):
# p = Pcoef @ [a0, a4, a8, a12]
_Pcoef = np.array([
    [1.0, 3.0, 105.0, 945.0],
    [0.0, -6.0, -420.0, -7560.0],
    [0.0, 1.0, 210.0, 13860.0],
    [0.0, 0.0, -28.0, -6930.0],
    [0.0, 0.0, 1.0, 1155.0],
    [0.0, 0.0, 0.0, -66.0],
    [0.0, 0.0, 0.0, 1.0]
])
# dp/da_c: for each c, dp_k/da_c = Pcoef[k, c] + Pcoef[k, 3] * da12[c]
_DP_DAC = np.zeros((3, 7))
for _c in range(3):
    for _k in range(7):
        _DP_DAC[_c, _k] = _Pcoef[_k, _c] + _Pcoef[_k, 3] * _da12[_c]

# Signs for q_k = p_k * (-1)^k
_SIGNS = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
# DQ_DAC[c,k] = DP_DAC[c,k] * (-1)^k
_DQ_DAC = _DP_DAC * _SIGNS[np.newaxis, :]

_TWOPi = 2.0 * np.pi


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # Coefficients of P(x) = sum_{k=0}^{6} p_k x^{2k}
    # p = Pcoef @ [a0, a4, a8, a12]
    p = _Pcoef @ np.array([a0, a4, a8, a12])
    
    # m_k = sum_{i,j} p_i p_j I[k+i+j] = p^T M_k p
    m0 = p @ _M_MAT[0] @ p
    m2 = p @ _M_MAT[1] @ p
    m4 = p @ _M_MAT[2] @ p
    
    # q_k = p_k * (-1)^k
    q = p * _SIGNS
    
    # n_k = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    n0 = _TWOPi * (q @ _M_MAT[0] @ q)
    n2 = _TWOPi * (q @ _M_MAT[1] @ q)
    n4 = _TWOPi * (q @ _M_MAT[2] @ q)
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Gradient: d(bound)/d(a_c) = (1/(2*bound)) * (vF * d(vf)/d(a_c) + vf * d(vF)/d(a_c))
    # d(vf)/d(a_c) = (dm2/da_c * m0 - m2 * dm0/da_c) / m0^2
    # dm_k/da_c = 2 * (M_k @ p) @ dp_c  where dp_c = DP_DAC[c,:]
    # Similarly for n
    
    # Precompute M_k @ p and M_k @ q for each k
    Mp0 = _M_MAT[0] @ p
    Mp1 = _M_MAT[1] @ p
    Mp2 = _M_MAT[2] @ p
    Mq0 = _M_MAT[0] @ q
    Mq1 = _M_MAT[1] @ q
    Mq2 = _M_MAT[2] @ q
    
    grad = np.zeros(3)
    for c in range(3):
        dp = _DP_DAC[c, :]
        dq = _DQ_DAC[c, :]
        
        dm0 = 2.0 * (dp @ Mp0)
        dm2 = 2.0 * (dp @ Mp1)
        dm4 = 2.0 * (dp @ Mp2)
        
        dn0 = 2.0 * _TWOPi * (dq @ Mq0)
        dn2 = 2.0 * _TWOPi * (dq @ Mq1)
        dn4 = 2.0 * _TWOPi * (dq @ Mq2)
        
        dvm2 = (dm2 * m0 - m2 * dm0) / (m0 * m0)
        dvn2 = (dn2 * n0 - n2 * dn0) / (n0 * n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def construct():
    rng = np.random.default_rng(42)
    
    # Phase 1: Coarse grid + random starts to find good basins
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-1.2, 0.06, 0.0005]),
        np.array([-1.0, 0.02, 0.0001]),
        np.array([-1.3, 0.03, 0.0002]),
        np.array([-0.8, 0.08, 0.001]),
    ]
    
    # Add random starts with varied scales
    for i in range(30):
        a0 = rng.uniform(-3.0, 1.0)
        a4 = rng.uniform(-0.1, 0.2)
        a8 = rng.uniform(-0.005, 0.005)
        starts.append(np.array([a0, a4, a8]))
    
    # Log-scale random starts for small coefficients
    for i in range(20):
        a0 = rng.uniform(-3.0, 1.0)
        log_a4 = rng.uniform(np.log(1e-4), np.log(0.2))
        log_a8 = rng.uniform(np.log(1e-6), np.log(0.005))
        sign4 = rng.choice([-1, 1])
        sign8 = rng.choice([-1, 1])
        starts.append(np.array([a0, sign4 * np.exp(log_a4), sign8 * np.exp(log_a8)]))
    
    best_val = np.inf
    best_params = None
    best_local = []  # Store multiple local minima
    
    for start in starts:
        if time.time() > _DEADLINE - 40:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-13})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
            if res.fun < 1e10 and np.isfinite(res.fun):
                best_local.append((res.fun, res.x.copy()))
        except Exception:
            pass
    
    # Phase 2: Basin hopping from the best local minima found
    best_local.sort(key=lambda x: x[0])
    for _, local in best_local[:5]:
        if time.time() > _DEADLINE - 20:
            break
        for step in range(10):
            if time.time() > _DEADLINE - 20:
                break
            # Perturb
            perturbation = rng.normal(0, 0.05, 3)
            perturbed = local + perturbation
            try:
                res = minimize(_objective_grad, perturbed, method='L-BFGS-B',
                              jac=True, options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-13})
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
    
    # Phase 3: Polish with Nelder-Mead
    if best_params is not None and time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final tight L-BFGS-B polish
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Ultra-tight final polish
    if best_params is not None and time.time() < _DEADLINE - 2:
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
