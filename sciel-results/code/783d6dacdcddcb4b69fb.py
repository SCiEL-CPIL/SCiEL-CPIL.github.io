"""
Combined program: exploits scale invariance to reduce to 1D search over ratio,
uses high-resolution grid + Brent's method for robust global minimization,
with analytic gradient polish for final refinement.
"""

import numpy as np
from scipy.optimize import minimize, brent
import time

# Precompute Gauss-Hermite nodes and weights
_N_GQ = 30
_X_GQ, _W_GQ = np.polynomial.hermite.hermgauss(_N_GQ)
_X2 = _X_GQ * _X_GQ

# Precompute Hermite polynomials H0, H4, H8, H12 at the nodes
_H0 = np.ones(_N_GQ)
_H4 = _X2**2 - 6 * _X2 + 3
_H8 = _X2**4 - 28 * _X2**3 + 210 * _X2**2 - 420 * _X2 + 105
_H12 = _X2**6 - 84 * _X2**5 + 1260 * _X2**4 - 12600 * _X2**3 + 55440 * _X2**2 - 100800 * _X2 + 10395

# Precompute derivatives
_X3 = _X_GQ**3
_X5 = _X_GQ**5
_X7 = _X_GQ**7
_X9 = _X_GQ**9
_X11 = _X_GQ**11

_DH4 = 4 * _X3 - 12 * _X_GQ
_DH8 = 8 * _X7 - 168 * _X5 + 840 * _X3 - 840 * _X_GQ
_DH12 = 12 * _X11 - 840 * _X9 + 10080 * _X7 - 75600 * _X5 + 221760 * _X3 - 201600 * _X_GQ

# Weights for integrals
_W_I0 = _W_GQ
_W_I2 = _W_GQ * _X2


def _compute_bound_3d(a, b, c):
    """
    Compute the uncertainty bound for (a, b, c).
    d is determined by P(0)=0 => a + 3b + 105c + 10395d = 0
    """
    d = -(a + 3 * b + 105 * c) / 10395.0
    
    P = a * _H0 + b * _H4 + c * _H8 + d * _H12
    I0 = np.dot(_W_I0, P * P)
    I2 = np.dot(_W_I2, P * P)
    
    P_prime = b * _DH4 + c * _DH8 + d * _DH12
    g_prime_poly = P_prime - _X_GQ * P
    J0 = np.dot(_W_I0, g_prime_poly * g_prime_poly)
    
    if I0 < 1e-15 or J0 < 1e-15 or I2 < 1e-15:
        return 1e10
    
    return np.sqrt(I2 * J0) / I0


def _compute_bound_and_grad_3d(a, b, c):
    """
    Compute the uncertainty bound and its gradient for (a, b, c).
    Returns (score, grad).
    """
    d = -(a + 3 * b + 105 * c) / 10395.0
    
    P = a * _H0 + b * _H4 + c * _H8 + d * _H12
    I0 = np.dot(_W_I0, P * P)
    I2 = np.dot(_W_I2, P * P)
    
    P_prime = b * _DH4 + c * _DH8 + d * _DH12
    g_prime_poly = P_prime - _X_GQ * P
    J0 = np.dot(_W_I0, g_prime_poly * g_prime_poly)
    
    if I0 < 1e-15 or J0 < 1e-15 or I2 < 1e-15:
        return 1e10, np.array([0.0, 0.0, 0.0])
    
    score = np.sqrt(I2 * J0) / I0
    
    # Gradient computation
    dP_da = _H0 - _H12 / 10395.0
    dP_db = _H4 - 3 * _H12 / 10395.0
    dP_dc = _H8 - 105 * _H12 / 10395.0
    
    dI0_da = 2 * np.dot(_W_I0, P * dP_da)
    dI0_db = 2 * np.dot(_W_I0, P * dP_db)
    dI0_dc = 2 * np.dot(_W_I0, P * dP_dc)
    
    dI2_da = 2 * np.dot(_W_I2, P * dP_da)
    dI2_db = 2 * np.dot(_W_I2, P * dP_db)
    dI2_dc = 2 * np.dot(_W_I2, P * dP_dc)
    
    dPp_da = -_DH12 / 10395.0
    dPp_db = _DH4 - 3 * _DH12 / 10395.0
    dPp_dc = _DH8 - 105 * _DH12 / 10395.0
    
    g_pp = P_prime - _X_GQ * P
    
    dgp_da = dPp_da - _X_GQ * dP_da
    dgp_db = dPp_db - _X_GQ * dP_db
    dgp_dc = dPp_dc - _X_GQ * dP_dc
    
    dJ0_da = 2 * np.dot(_W_I0, g_pp * dgp_da)
    dJ0_db = 2 * np.dot(_W_I0, g_pp * dgp_db)
    dJ0_dc = 2 * np.dot(_W_I0, g_pp * dgp_dc)
    
    grad_a = (dI2_da * J0 + I2 * dJ0_da) / (2.0 * I2 * J0) - dI0_da / I0
    grad_b = (dI2_db * J0 + I2 * dJ0_db) / (2.0 * I2 * J0) - dI0_db / I0
    grad_c = (dI2_dc * J0 + I2 * dJ0_dc) / (2.0 * I2 * J0) - dI0_dc / I0
    
    return score, np.array([grad_a, grad_b, grad_c])


def _bound_1d(t):
    """
    1D objective: set a=1, b=t, c=0 (or explore the full 2D manifold).
    Actually, we'll use the 2D reduction with a=1 and search over (b, c),
    but the key insight is that due to scale invariance, we can fix one parameter.
    
    For the 1D reduction: fix a=1, then we have (b, c) as free params.
    But we can further reduce by noting that the problem is homogeneous.
    
    Let's use the approach: fix a=1, search over (b, c) in 2D,
    but use a coarse-to-fine grid strategy with Brent for final polish.
    """
    b, c = t
    return _compute_bound_3d(1.0, b, c)


def _bound_1d_ratio(t):
    """
    True 1D reduction: fix a=1, b=1, search over c=t.
    This exploits the fact that we can normalize by scaling.
    """
    return _compute_bound_3d(1.0, 1.0, t)


def _bound_1d_ratio2(t):
    """
    Alternative 1D: fix a=0, b=1, search over c=t.
    """
    return _compute_bound_3d(0.0, 1.0, t)


def _grad_3d(params):
    """
    Gradient for 3D parameters.
    """
    _, grad = _compute_bound_and_grad_3d(params[0], params[1], params[2])
    return grad


def construct():
    start_time = time.time()
    deadline = 200.0  # Safety margin before 210s kill
    
    best_score = 1e10
    best_params = np.array([-113 / 100, 1 / 25, 1 / 3240])  # Seed
    
    # Evaluate seed
    seed_score = _compute_bound_3d(-113 / 100, 1 / 25, 1 / 3240)
    best_score = seed_score
    
    def _time_ok(margin=5.0):
        return time.time() < deadline - margin
    
    # Phase 1: 1D search over c with a=1, b=1
    # High-resolution grid
    if _time_ok(120):
        c_range = np.linspace(-10.0, 10.0, 2001)
        min_val = 1e10
        min_c = 0.0
        for c in c_range:
            if not _time_ok(120):
                break
            val = _bound_1d_ratio(c)
            if val < min_val:
                min_val = val
                min_c = c
        
        if min_val < best_score:
            best_score = min_val
            best_params = np.array([1.0, 1.0, min_c])
        
        # Brent's method refinement around the best grid point
        idx = np.argmin([_bound_1d_ratio(c) for c in [min_c - 0.01, min_c, min_c + 0.01]])
        c_left = min_c - 0.1
        c_right = min_c + 0.1
        try:
            c_opt = brent(_bound_1d_ratio, a=c_left, b=c_right, xtol=1e-12)
            val_opt = _bound_1d_ratio(c_opt)
            if val_opt < best_score:
                best_score = val_opt
                best_params = np.array([1.0, 1.0, c_opt])
        except Exception:
            pass
    
    # Phase 2: 1D search over c with a=0, b=1
    if _time_ok(100):
        c_range2 = np.linspace(-10.0, 10.0, 2001)
        min_val2 = 1e10
        min_c2 = 0.0
        for c in c_range2:
            if not _time_ok(100):
                break
            val = _bound_1d_ratio2(c)
            if val < min_val2:
                min_val2 = val
                min_c2 = c
        
        if min_val2 < best_score:
            best_score = min_val2
            best_params = np.array([0.0, 1.0, min_c2])
        
        c_left2 = min_c2 - 0.1
        c_right2 = min_c2 + 0.1
        try:
            c_opt2 = brent(_bound_1d_ratio2, a=c_left2, b=c_right2, xtol=1e-12)
            val_opt2 = _bound_1d_ratio2(c_opt2)
            if val_opt2 < best_score:
                best_score = val_opt2
                best_params = np.array([0.0, 1.0, c_opt2])
        except Exception:
            pass
    
    # Phase 3: 2D search with a=1, over (b, c)
    if _time_ok(60):
        # Coarse grid
        b_range = np.linspace(-5.0, 5.0, 51)
        c_range3 = np.linspace(-5.0, 5.0, 51)
        
        min_val3 = 1e10
        min_b3 = 0.0
        min_c3 = 0.0
        
        for b in b_range:
            if not _time_ok(60):
                break
            for c in c_range3:
                if not _time_ok(60):
                    break
                val = _compute_bound_3d(1.0, b, c)
                if val < min_val3:
                    min_val3 = val
                    min_b3 = b
                    min_c3 = c
        
        if min_val3 < best_score:
            best_score = min_val3
            best_params = np.array([1.0, min_b3, min_c3])
        
        # Refine with L-BFGS-B from the best grid point
        if _time_ok(30):
            try:
                result = minimize(
                    lambda x: _compute_bound_3d(x[0], x[1], x[2]),
                    np.array([1.0, min_b3, min_c3]),
                    jac=lambda x: _compute_bound_and_grad_3d(x[0], x[1], x[2])[1],
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if result.fun < best_score:
                    best_score = result.fun
                    best_params = result.x
            except Exception:
                pass
    
    # Phase 4: Full 3D polish from the best point found
    if _time_ok(10):
        try:
            result = minimize(
                lambda x: _compute_bound_3d(x[0], x[1], x[2]),
                best_params,
                jac=lambda x: _compute_bound_and_grad_3d(x[0], x[1], x[2])[1],
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-14}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Ensure we don't return worse than seed
    if best_score > seed_score:
        best_params = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    return best_params
