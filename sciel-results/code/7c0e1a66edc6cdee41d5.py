"""Analytic uncertainty bound with precomputed structure + multi-start optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 185  # tighter safety margin before 210s kill


def _hermite_coeffs():
    """
    Precompute the polynomial coefficients of H_0, H_4, H_8, H_12.
    H_n are physicists' Hermite polynomials.
    H_0 = 1
    H_4 = x^4 - 6x^2 + 3
    H_8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    H_12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    
    We represent P(x) = sum_{k=0}^{6} p_k * x^{2k}
    H_0: p = [1, 0, 0, 0, 0, 0, 0]
    H_4: p = [3, -6, 1, 0, 0, 0, 0]
    H_8: p = [105, -420, 210, -28, 1, 0, 0]
    H_12: p = [945, -7560, 13860, -6930, 1155, -66, 1]
    """
    H = np.zeros((4, 7))
    H[0, 0] = 1.0
    H[1, 0] = 3.0; H[1, 1] = -6.0; H[1, 2] = 1.0
    H[2, 0] = 105.0; H[2, 1] = -420.0; H[2, 2] = 210.0; H[2, 3] = -28.0; H[2, 4] = 1.0
    H[3, 0] = 945.0; H[3, 1] = -7560.0; H[3, 2] = 13860.0; H[3, 3] = -6930.0
    H[3, 4] = 1155.0; H[3, 5] = -66.0; H[3, 6] = 1.0
    return H


def _compute_moments():
    """
    Precompute I[n] = integral x^{2n} e^{-x^2} dx = (2n-1)!! * sqrt(pi) / 2^n
    for n = 0, 1, ..., 12 (need up to x^{24} since max degree is 12, |P|^2 has degree 24)
    """
    I = np.zeros(13)
    sp = np.sqrt(np.pi)
    for n in range(13):
        if n == 0:
            I[n] = sp
        else:
            # Recurrence: I[n] = I[n-1] * (2n-1) / (4*n)
            I[n] = I[n-1] * (2.0*n - 1.0) / (4.0 * n)
    return I


_H = _hermite_coeffs()
_I = _compute_moments()
_SIGN = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])


def _bound(params):
    """
    Compute A(f)*A(F) for g(x) = P(x) * exp(-x^2/2)
    where P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12 with a12 = -(a0+3a4+105a8)/945
    Vectorized inner loops for speed.
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    coeffs = np.array([a0, a4, a8, a12])
    p = coeffs @ _H  # shape (7,)
    
    # Vectorized: build index matrix and use matrix operations
    # m0 = sum_{i,j} p_i p_j I[i+j], m2 = sum_{i,j} p_i p_j I[1+i+j]
    idx = np.arange(7)
    I_mat0 = _I[idx[:,None] + idx[None,:]]  # 7x7 matrix of I[i+j]
    I_mat2 = _I[1 + idx[:,None] + idx[None,:]]  # 7x7 matrix of I[1+i+j]
    
    m0 = p @ I_mat0 @ p
    m2 = p @ I_mat2 @ p
    
    # Q coefficients: q_k = p_k * (-1)^k
    q = p * _SIGN
    
    n0 = 2.0 * np.pi * (q @ I_mat0 @ q)
    n2 = 2.0 * np.pi * (q @ I_mat2 @ q)
    
    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18
    
    return np.sqrt(vf * vF)


def _bound_grad(params):
    """Compute bound and its gradient analytically using precomputed structures. Vectorized."""
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    coeffs = np.array([a0, a4, a8, a12])
    p = coeffs @ _H
    
    # Precompute index matrix once
    idx = np.arange(7)
    I_mat0 = _I[idx[:,None] + idx[None,:]]
    I_mat2 = _I[1 + idx[:,None] + idx[None,:]]
    
    # Compute m0, m2 vectorized
    m0 = p @ I_mat0 @ p
    m2 = p @ I_mat2 @ p
    
    q = p * _SIGN
    
    n0 = 2.0 * np.pi * (q @ I_mat0 @ q)
    n2 = 2.0 * np.pi * (q @ I_mat2 @ q)
    
    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Gradient: d(bound)/da_c = (vF * d(vf)/da_c + vf * d(vF)/da_c) / (2*bound)
    # d(vf)/da_c = (dm2/da_c * m0 - m2 * dm0/da_c) / m0^2
    
    # da12/da_c for c = a0, a4, a8:
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    # dp_k/da_c = H[c,k] + da12[c] * H[3,k]
    grad = np.zeros(3)
    for c in range(3):
        dp = _H[c, :] + da12[c] * _H[3, :]
        
        # dm0/da_c = 2 * dp^T I_mat0 p
        # dm2/da_c = 2 * dp^T I_mat2 p
        dm0 = 2.0 * (dp @ I_mat0 @ p)
        dm2 = 2.0 * (dp @ I_mat2 @ p)
        
        dq = dp * _SIGN
        
        # dn0/da_c = 2*pi * 2 * dq^T I_mat0 q = 4*pi * dq^T I_mat0 q
        # dn2/da_c = 4*pi * dq^T I_mat2 q
        dn0 = 4.0 * np.pi * (dq @ I_mat0 @ q)
        dn2 = 4.0 * np.pi * (dq @ I_mat2 @ q)
        
        dvm = (dm2 * m0 - m2 * dm0) / (m0 * m0)
        dvn = (dn2 * n0 - n2 * dn0) / (n0 * n0)
        
        grad[c] = (vF * dvm + vf * dvn) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    return _bound(params)


def _objective_grad(params):
    return _bound_grad(params)


def construct():
    """
    Multi-start optimization with fast precomputed evaluation.
    """
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642])
    
    # Starting points: seed + diverse structured exploration
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),   # seed
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
        np.array([-1.3, 0.03, 0.00025]),
        np.array([-0.9, 0.07, 0.0009]),
        np.array([-1.7, 0.02, 0.00015]),
        # Additional diverse starts
        np.array([-1.13, 0.035, 0.00028]),
        np.array([-1.13, 0.045, 0.00033]),
        np.array([-1.08, 0.04, 0.00032]),
        np.array([-1.18, 0.04, 0.00030]),
        np.array([-1.13, 0.04, 0.00028]),
        np.array([-1.13, 0.04, 0.00034]),
    ]
    
    # Add random starts with diverse seeds
    rng = np.random.default_rng(42)
    for _ in range(30):
        starts.append(rng.normal([0, 0.04, 0.0005], [1.0, 0.1, 0.002]))
    
    # Phase 1: L-BFGS-B from all starts
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Polish with Nelder-Mead from best
    if best_params is not None and time.time() < _DEADLINE - 25:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 30000, 'xatol': 1e-16, 'fatol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final L-BFGS-B polish with tighter tolerances
    if best_params is not None and time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-17, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Coordinate descent refinement for final polish
    if best_params is not None and time.time() < _DEADLINE - 15:
        # Try small perturbations around best with coordinate descent
        for scale in [0.05, 0.01, 0.001]:
            if time.time() > _DEADLINE - 5:
                break
            for _ in range(5):
                perturbed = best_params + scale * rng.normal(0, 1, 3)
                try:
                    res = minimize(_objective_grad, perturbed, method='L-BFGS-B',
                                  jac=True, options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-17})
                    if res.fun < best_val and np.isfinite(res.fun):
                        best_val = res.fun
                        best_params = res.x.copy()
                except Exception:
                    pass
    
    # Phase 5: High-precision final polish with very tight tolerances
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 20000, 'ftol': 1e-20, 'gtol': 1e-19, 'maxls': 100})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
        
        # One more Nelder-Mead polish at extreme precision
        if time.time() < _DEADLINE - 3:
            try:
                res = minimize(_objective, best_params, method='Nelder-Mead',
                              options={'maxiter': 50000, 'xatol': 1e-18, 'fatol': 1e-20, 'adaptive': True})
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
