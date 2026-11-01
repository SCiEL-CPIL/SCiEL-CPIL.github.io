"""Analytic uncertainty bound with exact gradient + L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200

# Precompute constants for faster evaluation
# I[n] = integral x^{2n} e^{-x^2} dx
# I[n] = (2n)! / (4^n n!) * sqrt(pi)
_I = np.zeros(14)
_sp = np.sqrt(np.pi)
for n in range(14):
    val = _sp
    for m in range(1, 2*n+1):
        val *= m
    for m in range(1, n+1):
        val /= (4.0 * m)
    _I[n] = val

# Matrix M such that p = M * a
# p_0 = 0
# p_1 = 8 a0 + 18 a4 + 420 a8
# p_2 = -44/3 a0 + (1 - 44) a4 + (210 - 44*105/3) a8  -> 210 - 1540 = -1330
# p_3 = -28 a8 - 6930 a12 = -28 a8 + 7.333... (a0+3a4+105a8) -> 6930/945 = 22/3
#      p3 = 22/3 a0 + 22 a4 + (-28 + 22*105/3) a8 = 22/3 a0 + 22 a4 + (-28 + 770) a8 = 22/3 a0 + 22 a4 + 742 a8
# p_4 = a8 + 1155 a12 = a8 - 1.222... (a0+3a4+105a8) -> 1155/945 = 11/9
#      p4 = -11/9 a0 - 11/3 a4 + (1 - 11/9*105) a8 = -11/9 a0 - 11/3 a4 + (9/9 - 1155/9) a8 = -11/9 a0 - 11/3 a4 - 1146/9 a8 = -11/9 a0 - 11/3 a4 - 382/3 a8
# p_5 = -66 a12 = 66/945 (a0+3a4+105a8) = 22/315 (a0+3a4+105a8)
#      p5 = 22/315 a0 + 66/315 a4 + 2310/315 a8 = 22/315 a0 + 22/105 a4 + 22/3 a8
# p_6 = a12 = -1/945 a0 - 3/945 a4 - 105/945 a8 = -1/945 a0 - 1/315 a4 - 1/9 a8

_M = np.array([
    [0.0, 0.0, 0.0],
    [8.0, 18.0, 420.0],
    [-44.0/3.0, 1.0 - 44.0, 210.0 - 44.0*105.0/3.0],
    [22.0/3.0, 22.0, -28.0 + 22.0*105.0/3.0],
    [-11.0/9.0, -11.0/3.0, 1.0 - 11.0*105.0/9.0],
    [22.0/315.0, 66.0/315.0, 2310.0/315.0],
    [-1.0/945.0, -1.0/315.0, -1.0/9.0]
])

# S_k matrices for moments
# m_k = p^T S_k p, where (S_k)_{ij} = I[k+i+j]
_S = np.zeros((3, 7, 7))
for k in range(3):
    for i in range(7):
        for j in range(7):
            idx = k + i + j
            if idx < 14:
                _S[k, i, j] = _I[idx]

# Q_k = M^T S_k M
_Q = np.zeros((3, 3, 3))
for k in range(3):
    _Q[k] = _M.T @ _S[k] @ _M

# For Fourier: q_k = p_k (-1)^k
# Let D be diagonal matrix with D[k,k] = (-1)^k
_D = np.diag([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
_Mq = _D @ _M
_Sq = _S # Same I values
_Qq = np.zeros((3, 3, 3))
for k in range(3):
    _Qq[k] = _Mq.T @ _Sq[k] @ _Mq

# Note: n_k = 2*pi * q^T S_k q = 2*pi * a^T _Qq[k] a
_factor_n = 2.0 * np.pi


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    """
    a = params
    a0, a4, a8 = a[0], a[1], a[2]
    
    # m_k = a^T Q_k a
    m0 = a @ _Q[0] @ a
    m2 = a @ _Q[1] @ a
    # m4 = a @ _Q[2] @ a (not needed for bound, but good for check)
    
    # n_k = 2*pi * a^T Qq_k a
    n0 = _factor_n * (a @ _Qq[0] @ a)
    n2 = _factor_n * (a @ _Qq[1] @ a)
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Gradient
    # d(vf)/da = (2 * Q1 @ a * m0 - m2 * 2 * Q0 @ a) / m0^2
    # d(vF)/da = (2 * Qq1 @ a * n0 - n2 * 2 * Qq0 @ a) / n0^2
    # d(bound)/da = 1/(2*bound) * (vF * d(vf)/da + vf * d(vF)/da)
    
    Q0_a = _Q[0] @ a
    Q1_a = _Q[1] @ a
    Qq0_a = _Qq[0] @ a
    Qq1_a = _Qq[1] @ a
    
    d_vf = (2.0 * Q1_a * m0 - m2 * 2.0 * Q0_a) / (m0 * m0)
    d_vF = (2.0 * Qq1_a * n0 - n2 * 2.0 * Qq0_a) / (n0 * n0)
    
    grad = (vF * d_vf + vf * d_vF) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def construct():
    # Multiple starting points
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
    ]
    
    rng = np.random.default_rng(123)
    for i in range(20):
        starts.append(rng.normal(0, 0.5, 3))
    
    best_val = np.inf
    best_params = None
    
    for start in starts:
        if time.time() > _DEADLINE - 20:
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
    if best_params is not None and time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final polish with L-BFGS-B
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
