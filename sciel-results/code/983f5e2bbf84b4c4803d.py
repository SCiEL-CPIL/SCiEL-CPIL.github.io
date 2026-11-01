"""Combined: precomputed quadratic forms + multi-start L-BFGS-B + grid refinement + high-precision polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 195  # safety margin before 210s kill


def _hermite_coeffs():
    H = np.zeros((4, 7))
    H[0, 0] = 1.0
    H[1, 0] = 3.0; H[1, 1] = -6.0; H[1, 2] = 1.0
    H[2, 0] = 105.0; H[2, 1] = -420.0; H[2, 2] = 210.0; H[2, 3] = -28.0; H[2, 4] = 1.0
    H[3, 0] = 945.0; H[3, 1] = -7560.0; H[3, 2] = 13860.0; H[3, 3] = -6930.0
    H[3, 4] = 1155.0; H[3, 5] = -66.0; H[3, 6] = 1.0
    return H


def _compute_moments():
    I = np.zeros(13)
    sp = np.sqrt(np.pi)
    I[0] = sp
    for n in range(1, 13):
        I[n] = I[n-1] * (2.0*n - 1.0) / (4.0 * n)
    return I


_H = _hermite_coeffs()
_I = _compute_moments()

# Precompute symmetric quadratic-form matrices (7x7)
# m_k = p^T M_k p,  n_k = 2*pi * q^T N_k q
_M = np.zeros((3, 7, 7))
_N = np.zeros((3, 7, 7))
for k in range(3):
    for i in range(7):
        for j in range(7):
            _M[k, i, j] = 0.5 * _I[k + i + j]
            _N[k, i, j] = 0.5 * _I[k + i + j]

_SIGNS = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
# a12 = -(a0 + 3*a4 + 105*a8)/945
_DA12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
# Effective coefficient matrix for the 3 free params:
# p = C @ a  where C[c, k] = H[c, k] + da12[c] * H[3, k]
_C = _H[:3, :] + np.outer(_DA12, _H[3, :])


def _compute_all(a):
    """a: shape (3,). Returns m0,m2,n0,n2, p, q."""
    p = _C @ a
    q = p * _SIGNS
    m0 = p @ _M[0] @ p
    m2 = p @ _M[1] @ p
    n0 = 2.0 * np.pi * (q @ _N[0] @ q)
    n2 = 2.0 * np.pi * (q @ _N[1] @ q)
    return m0, m2, n0, n2, p, q


def _bound(a):
    m0, m2, n0, n2, _, _ = _compute_all(a)
    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
    vf = m2 / m0
    vF = n2 / n0
    if vf <= 0 or vF <= 0:
        return 1e18
    return np.sqrt(vf * vF)


def _bound_grad(a):
    m0, m2, n0, n2, p, q = _compute_all(a)
    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    vf = m2 / m0
    vF = n2 / n0
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    bound = np.sqrt(vf * vF)

    # dm_k/da_c = 2 * (M_k p)_c  for the free-parameter representation
    # since m_k = (C a)^T M_k (C a), d/da_c = 2 (C^T M_k C a)_c
    # But C a = p, and C^T M_k C = effective matrix
    # Simpler: dm_k/da_c = 2 * (M_k p) @ C[c, :]
    grad = np.zeros(3)
    Mp0 = _M[0] @ p
    Mp1 = _M[1] @ p
    Nq0 = _N[0] @ q
    Nq1 = _N[1] @ q
    for c in range(3):
        dm0 = 2.0 * np.dot(Mp0, _C[c, :])
        dm2 = 2.0 * np.dot(Mp1, _C[c, :])
        dq = _C[c, :] * _SIGNS
        dn0 = 2.0 * 2.0 * np.pi * np.dot(Nq0, dq)
        dn2 = 2.0 * 2.0 * np.pi * np.dot(Nq1, dq)
        dvm = (dm2 * m0 - m2 * dm0) / (m0 * m0)
        dvn = (dn2 * n0 - n2 * dn0) / (n0 * n0)
        grad[c] = (vF * dvm + vf * dvn) / (2.0 * bound)
    return bound, grad


def _objective(a):
    return _bound(a)


def _objective_grad(a):
    return _bound_grad(a)


def construct():
    """Multi-start L-BFGS-B + grid refinement + high-precision polish."""
    from scipy.optimize import minimize

    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642])

    # Structured + random starting points
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-1.5, 0.03, 0.0002]),
        np.array([-0.8, 0.06, 0.0008]),
        np.array([-2.0, 0.02, 0.0001]),
        np.array([-1.2, 0.035, 0.0003]),
        np.array([-1.1, 0.045, 0.00035]),
        np.array([-0.5, 0.08, 0.001]),
        np.array([-3.0, 0.01, 0.00005]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, -0.05, 0.001]),
        np.array([-2.5, 0.015, 0.00015]),
        np.array([-1.15, 0.042, 0.00032]),
        np.array([-1.05, 0.038, 0.00028]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([0.5, 0.02, 0.0002]),
    ]

    rng = np.random.default_rng(42)
    for _ in range(30):
        starts.append(rng.normal([0, 0.04, 0.0005], [1.0, 0.1, 0.002]))

    for start in starts:
        if time.time() > _DEADLINE - 25:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 100})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Grid refinement around best
    if time.time() < _DEADLINE - 30:
        center = best_params.copy()
        scales = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
        for scale in scales:
            if time.time() > _DEADLINE - 15:
                break
            for axis in range(3):
                for direction in [-1, 1]:
                    if time.time() > _DEADLINE - 10:
                        break
                    trial = center.copy()
                    trial[axis] += direction * scale
                    try:
                        val = _bound(trial)
                        if val < best_val:
                            best_val = val
                            best_params = trial.copy()
                            res = minimize(_objective_grad, trial, method='L-BFGS-B',
                                          jac=True, options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16})
                            if res.fun < best_val and np.isfinite(res.fun):
                                best_val = res.fun
                                best_params = res.x.copy()
                    except Exception:
                        pass

    # High-precision polish: Trust-Constr with tighter tolerances
    if time.time() < _DEADLINE - 8:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 50000, 'ftol': 1e-22, 'gtol': 1e-20, 'maxls': 200})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Nelder-Mead final polish (derivative-free, can escape tiny numerical plateaus)
    if time.time() < _DEADLINE - 4:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 100000, 'xatol': 1e-18, 'fatol': 1e-22, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # One more L-BFGS-B from the Nelder-Mead result if it improved
    if time.time() < _DEADLINE - 2:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 20000, 'ftol': 1e-22, 'gtol': 1e-20})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    return np.array(best_params)
# EVOLVE-BLOCK-END
