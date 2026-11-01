"""Combined uncertainty bound optimization with analytic gradient."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195

# Precompute I[n] = integral x^{2n} e^{-x^2} dx
def _compute_I_array(max_n=14):
    I = np.zeros(max_n + 1)
    I[0] = np.sqrt(np.pi)
    for n in range(1, max_n + 1):
        I[n] = I[n-1] * (2*n - 1) / (2*n)
    return I

_I = _compute_I_array(14)
_TWO_PI = 2.0 * np.pi

# Hermite polynomial coefficients in basis {1, x^2, x^4, x^6, x^8, x^10, x^12}
_H_COEFFS = np.array([
    [1, 0, 0, 0, 0, 0, 0],
    [3, -6, 1, 0, 0, 0, 0],
    [105, -420, 210, -28, 1, 0, 0],
    [945, -7560, 13860, -6930, 1155, -66, 1]
], dtype=float)

_NEG_ONE_POW = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

# Precompute Gram matrices for moments
# M_k = matrix where M_k[i,j] = I[k+i+j] for k=0,1,2
# N_k = 2*pi * matrix where N_k[i,j] = I[k+i+j] * (-1)^{i+j}
_N = 7
_M = np.zeros((3, _N, _N))
_N_mat = np.zeros((3, _N, _N))
for k in range(3):
    for i in range(_N):
        for j in range(_N):
            idx = k + i + j
            if idx <= 14:
                _M[k, i, j] = _I[idx]
                _N_mat[k, i, j] = _TWO_PI * _I[idx] * _NEG_ONE_POW[i] * _NEG_ONE_POW[j]


def _compute_bound_and_grad(params):
    """
    Compute uncertainty bound sqrt(vf * vF) and its gradient using vectorized ops.
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    coeffs = np.array([a0, a4, a8, a12])
    p = _H_COEFFS @ coeffs
    q = p * _NEG_ONE_POW
    
    # m_k = p^T M_k p, n_k = q^T N_k q
    m0 = p @ _M[0] @ p
    m1 = p @ _M[1] @ p
    n0 = q @ _N_mat[0] @ q
    n1 = q @ _N_mat[1] @ q
    
    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m1 / m0
    vF = n1 / n0
    
    if vf <= 0 or vF <= 0 or not np.isfinite(vf) or not np.isfinite(vF):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    if not np.isfinite(bound):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    # Gradient: dp/da_c = _H_COEFFS[:, c] + da12_coeffs[c] * _H_COEFFS[:, 3]
    da12_coeffs = np.array([-1.0, -3.0, -105.0]) / 945.0
    grad = np.zeros(3)
    
    for c in range(3):
        dp = _H_COEFFS[:, c] + da12_coeffs[c] * _H_COEFFS[:, 3]
        dq = dp * _NEG_ONE_POW
        
        dm0 = 2.0 * p @ _M[0] @ dp
        dm1 = 2.0 * p @ _M[1] @ dp
        dn0 = 2.0 * q @ _N_mat[0] @ dq
        dn1 = 2.0 * q @ _N_mat[1] @ dq
        
        dvm2 = (dm1 * m0 - m1 * dm0) / (m0 * m0)
        dvn2 = (dn1 * n0 - n1 * dn0) / (n0 * n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    val, _ = _compute_bound_and_grad(params)
    return val


def _objective_grad(params):
    return _compute_bound_and_grad(params)


def construct():
    start_time = time.time()
    
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-1.5, 0.06, 0.0006]),
        np.array([-2.0, 0.08, 0.0008]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-2.5, 0.1, 0.001]),
        np.array([-3.0, 0.12, 0.0012]),
        np.array([0.5, -0.03, 0.0003]),
        np.array([-1.2, 0.045, 0.00035]),
        np.array([-1.13, 0.041, 0.00031]),
        np.array([-1.12, 0.039, 0.00030]),
        np.array([-1.14, 0.042, 0.00032]),
    ]
    
    rng = np.random.default_rng(42)
    for _ in range(20):
        starts.append(rng.normal(0, 0.8, 3))
    
    best_val = np.inf
    best_params = None
    
    # Phase 1: L-BFGS-B with analytic gradient from multiple starts
    for start in starts:
        if time.time() - start_time > _DEADLINE - 40:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Trust-region refinement from best
    if best_params is not None and time.time() - start_time < _DEADLINE - 20:
        try:
            res = minimize(_objective_grad, best_params, method='trust-constr',
                          jac=True, options={'maxiter': 2000, 'gtol': 1e-12, 'xtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead polish
    if best_params is not None and time.time() - start_time < _DEADLINE - 10:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 100000, 'xatol': 1e-14, 'fatol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision L-BFGS-B polish
    if best_params is not None and time.time() - start_time < _DEADLINE - 3:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 20000, 'ftol': 1e-17, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        best_params = np.array([-1.13, 0.04, 0.000308642])
    
    return best_params
# EVOLVE-BLOCK-END
