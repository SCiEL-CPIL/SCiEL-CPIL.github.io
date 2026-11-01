"""Analytic uncertainty bound with vectorized evaluation + multi-start optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200


def _precompute():
    """Precompute I[n] = integral x^(2n) e^{-x^2} dx for n=0..13."""
    sp = np.sqrt(np.pi)
    I = np.zeros(14)
    for n in range(14):
        # (2n)! / (4^n * n!) * sqrt(pi)
        if n == 0:
            I[n] = sp
        else:
            # Recurrence: I[n] = I[n-1] * (2n-1) / (4n)
            I[n] = I[n-1] * (2*n - 1) / (4*n)
    return I


_I = _precompute()

# Coefficient matrices for polynomial P(x) = sum_{k=0}^{6} p_k x^{2k}
# p = M * [a0, a4, a8, a12]^T
_M = np.array([
    [1.0,    3.0,   105.0,    945.0],
    [0.0,   -6.0,  -420.0,  -7560.0],
    [0.0,    1.0,   210.0,  13860.0],
    [0.0,    0.0,   -28.0,  -6930.0],
    [0.0,    0.0,     1.0,   1155.0],
    [0.0,    0.0,     0.0,    -66.0],
    [0.0,    0.0,     0.0,     1.0],
], dtype=np.float64)

# a12 = -(a0 + 3*a4 + 105*a8) / 945
# So p = M[:, :3] * [a0,a4,a8] + M[:, 3:4] * a12
_M3 = _M[:, :3]
_M12 = _M[:, 3:4]
_da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

# q_k = p_k * (-1)^k
_signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

# For gradient: dp/da_c for each c in {0,1,2}
# dp_k/da_c = M[k,c] + M[k,3] * da12[c]
_dp_da = np.zeros((3, 7))
for c in range(3):
    _dp_da[c] = _M3[:, c] + _M12[:, 0] * _da12[c]

# Precompute S[k, i] = sum_j p_j * I[k+i+j] ... but p varies, so can't precompute.
# Instead, precompute the structure: for moment m_k = sum_{i,j} p_i p_j I[k+i+j]
# We can write m_k = p^T A_k p where A_k[i,j] = I[k+i+j]
_A = np.zeros((3, 7, 7))
for k in range(3):
    for i in range(7):
        for j in range(7):
            idx = k + i + j
            if idx < 14:
                _A[k, i, j] = _I[idx]

# For Fourier: n_k = 2*pi * q^T A_k q.
# For k=0,4,8,12, i^k=1, so Q(xi) = P(xi) in Hermite basis, implying power coeffs q = p.
# Thus n_k = 2*pi * p^T A_k p. We reuse _A.
_Aq = _A

# Gradient precomputation:
# dm_k/da_c = 2 * sum_i dp_i * (sum_j p_j A_k[i,j])
# = 2 * dp^T (A_k p)
# So if we let r_k = A_k p (vector of length 7), then dm_k = 2 * dp^T r_k
# Similarly dn_k = 4*pi * dq^T (A_k q) = 4*pi * (dp * signs)^T (A_k * (p * signs))
# Note A_k * (p*signs) = (signs * A_k * signs) * p = Aq_k * p
# So dn_k = 4*pi * (dp * signs)^T (Aq_k * p)


def _bound_and_grad(params):
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    p = _M3 @ params + _M12[0, 0] * a12
    
    # Moments
    m0 = p @ _A[0] @ p
    m2 = p @ _A[1] @ p
    m4 = p @ _A[2] @ p
    
    n0 = 2.0 * np.pi * (p @ _A[0] @ p)
    n2 = 2.0 * np.pi * (p @ _A[1] @ p)
    n4 = 2.0 * np.pi * (p @ _A[2] @ p)
    
    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Gradient
    grad = np.zeros(3)
    # Precompute r_k = A_k p outside loop
    r0 = _A[0] @ p
    r1 = _A[1] @ p
    r2 = _A[2] @ p
    
    for c in range(3):
        dp = _dp_da[c]
        
        dm0 = 2.0 * np.dot(dp, r0)
        dm2 = 2.0 * np.dot(dp, r1)
        dm4 = 2.0 * np.dot(dp, r2)
        
        # rq is same as r since Aq=A
        dn0 = 4.0 * np.pi * np.dot(dp, r0)
        dn2 = 4.0 * np.pi * np.dot(dp, r1)
        dn4 = 4.0 * np.pi * np.dot(dp, r2)
        
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
    # Seed value
    seed_params = np.array([-1.13, 0.04, 0.000308642])
    seed_val, _ = _bound_and_grad(seed_params)
    best_val = seed_val
    best_params = seed_params.copy()
    
    # Multi-start points
    starts = [
        seed_params,
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-2.0, 0.08, 0.0008]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.06, 0.0006]),
        np.array([0.5, -0.03, 0.0003]),
        np.array([-3.0, 0.12, 0.0012]),
        np.array([-0.8, 0.03, 0.0003]),
        np.array([-1.2, 0.035, 0.00035]),
    ]
    
    rng = np.random.default_rng(2024)
    # Add random starts around seed
    for i in range(15):
        starts.append(seed_params + rng.normal(0, 0.3, 3))
    # Add broader random starts
    for i in range(10):
        starts.append(rng.normal(0, 1.0, 3))
    
    # Phase 1: L-BFGS-B from multiple starts
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead polish
    if best_params is not None and time.time() < _DEADLINE - 20:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: High-precision L-BFGS-B polish
    if best_params is not None and time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-17})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final tight polish
    if best_params is not None and time.time() < _DEADLINE - 8:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-20, 'gtol': 1e-19})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Nelder-Mead final polish
    if best_params is not None and time.time() < _DEADLINE - 3:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-15, 'fatol': 1e-17})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None or not np.all(np.isfinite(best_params)):
        return seed_params
    
    return np.array(best_params, dtype=np.float64)
# EVOLVE-BLOCK-END
