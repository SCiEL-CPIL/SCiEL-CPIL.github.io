"""Combined program: High-precision quadrature + Global DE + Local Polish."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_hermite
import time

_DEADLINE = 175  # Safety margin before 210s kill

def _hermite_h(n, x):
    """Compute physicists' Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    elif n == 2:
        return 4 * x**2 - 2
    elif n == 3:
        return 8 * x**3 - 12 * x
    elif n == 4:
        return 16 * x**4 - 48 * x**2 + 12
    elif n == 7:
        return 128*x**7 - 1344*x**5 + 3360*x**3 - 1680*x
    elif n == 8:
        return 256*x**8 - 4480*x**6 + 26880*x**4 - 67200*x**2 + 40320
    elif n == 11:
        return (2048*x**11 - 77824*x**9 + 887040*x**7 
                - 4084080*x**5 + 8316000*x**3 - 6652800*x)
    elif n == 12:
        return (4096*x**12 - 147456*x**10 + 2580480*x**8 
                - 20321280*x**6 + 81729600*x**4 
                - 172972800*x**2 + 103219200)
    else:
        raise ValueError(f"Hermite H_{n} not implemented")
    """Compute physicists' Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    elif n == 2:
        return 4 * x**2 - 2
    elif n == 3:
        return 8 * x**3 - 12 * x
    elif n == 4:
        return 16 * x**4 - 48 * x**2 + 12
    elif n == 7:
        # H_7(x) = 128x^7 - 1344x^5 + 3360x^3 - 1680x
        return 128*x**7 - 1344*x**5 + 3360*x**3 - 1680*x
    elif n == 8:
        return 256*x**8 - 4480*x**6 + 26880*x**4 - 67200*x**2 + 40320
    elif n == 11:
        # H_11(x) = 2048x^11 - 77824x^9 + 887040x^7 - 4084080x^5 + 8316000x^3 - 6652800x
        return (2048*x**11 - 77824*x**9 + 887040*x**7 
                - 4084080*x**5 + 8316000*x**3 - 6652800*x)
    elif n == 12:
        return (4096*x**12 - 147456*x**10 + 2580480*x**8 
                - 20321280*x**6 + 81729600*x**4 
                - 172972800*x**2 + 103219200)
    else:
        raise ValueError(f"Hermite H_{n} not implemented")

# Precompute quadrature data for fast evaluation
_quad_data = None
_N_QUAD = 200  # Base precision

def _get_quad_data():
    global _quad_data
    if _quad_data is None:
        x_quad, w_quad = roots_hermite(_N_QUAD)
        
        H0 = _hermite_h(0, x_quad)
        H4 = _hermite_h(4, x_quad)
        H8 = _hermite_h(8, x_quad)
        H12 = _hermite_h(12, x_quad)
        
        dH0 = np.zeros_like(x_quad)
        dH4 = 8 * _hermite_h(3, x_quad)
        dH8 = 16 * _hermite_h(7, x_quad)
        dH12 = 24 * _hermite_h(11, x_quad)
        
        _quad_data = (x_quad, w_quad, H0, H4, H8, H12, dH0, dH4, dH8, dH12)
    return _quad_data

# High-precision quadrature for final polish
_quad_data_hp = None
_N_QUAD_HP = 400

def _get_quad_data_hp():
    global _quad_data_hp
    if _quad_data_hp is None:
        x_quad, w_quad = roots_hermite(_N_QUAD_HP)
        
        H0 = _hermite_h(0, x_quad)
        H4 = _hermite_h(4, x_quad)
        H8 = _hermite_h(8, x_quad)
        H12 = _hermite_h(12, x_quad)
        
        dH0 = np.zeros_like(x_quad)
        dH4 = 8 * _hermite_h(3, x_quad)
        dH8 = 16 * _hermite_h(7, x_quad)
        dH12 = 24 * _hermite_h(11, x_quad)
        
        _quad_data_hp = (x_quad, w_quad, H0, H4, H8, H12, dH0, dH4, dH8, dH12)
    return _quad_data_hp

def _compute_bound(coeffs, use_hp=False):
    """
    Compute the upper bound on C4 (uncertainty product) given coefficients [a0, a4, a8].
    """
    if time.time() > _DEADLINE:
        return float('inf')
    
    a0, a4, a8 = coeffs
    
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 40320.0
    H12_0 = 103219200.0
    
    d = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    if use_hp:
        x_quad, w_quad, H0, H4, H8, H12, dH0, dH4, dH8, dH12 = _get_quad_data_hp()
    else:
        x_quad, w_quad, H0, H4, H8, H12, dH0, dH4, dH8, dH12 = _get_quad_data()
    
    P = a0 * H0 + a4 * H4 + a8 * H8 + d * H12
    dP = a0 * dH0 + a4 * dH4 + a8 * dH8 + d * dH12
    
    P2 = P * P
    norm2 = np.sum(w_quad * P2)
    
    if norm2 <= 1e-15 or not np.isfinite(norm2):
        return 1e10
        
    var_x = np.sum(w_quad * x_quad**2 * P2) / norm2
    
    deriv_term = dP - x_quad * P
    deriv_term_sq = deriv_term * deriv_term
    norm_deriv2 = np.sum(w_quad * deriv_term_sq)
    
    if not np.isfinite(norm_deriv2) or norm_deriv2 < 0:
        return 1e10

    var_k = norm_deriv2 / norm2
    
    bound = var_x * var_k
    return bound

def construct():
    start_time = time.time()
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_result = seed.copy()
    best_score = _compute_bound(seed, use_hp=False)
    
    # 1. Global Search using Differential Evolution with adaptive bounds
    # Start with wider bounds, then refine
    bounds_wide = [(-2.0, 2.0), (-0.5, 0.5), (-0.05, 0.05)]
    bounds_narrow = [(-1.0, 1.0), (-0.1, 0.1), (-0.01, 0.01)]
    
    # First DE run with wider bounds
    if time.time() < _DEADLINE - 80:
        try:
            de_res = differential_evolution(
                lambda x: _compute_bound(x, use_hp=False), 
                bounds_wide, 
                seed=42, 
                maxiter=80, 
                tol=1e-6,
                popsize=12,
                polish=False
            )
            if de_res.fun < best_score:
                best_score = de_res.fun
                best_result = de_res.x.copy()
        except Exception:
            pass
    
    # Second DE run with narrower bounds around best found
    if time.time() < _DEADLINE - 50:
        try:
            center = best_result
            bounds_narrow = [
                (center[0] - 0.5, center[0] + 0.5),
                (center[1] - 0.05, center[1] + 0.05),
                (center[2] - 0.005, center[2] + 0.005)
            ]
            de_res2 = differential_evolution(
                lambda x: _compute_bound(x, use_hp=False), 
                bounds_narrow, 
                seed=123, 
                maxiter=60, 
                tol=1e-8,
                popsize=10,
                polish=False
            )
            if de_res2.fun < best_score:
                best_score = de_res2.fun
                best_result = de_res2.x.copy()
        except Exception:
            pass
    
    # 2. Local Polish using Nelder-Mead with medium precision
    if time.time() < _DEADLINE - 30:
        try:
            nm_res = minimize(
                lambda x: _compute_bound(x, use_hp=False), 
                best_result, 
                method='Nelder-Mead', 
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if nm_res.fun < best_score:
                best_score = nm_res.fun
                best_result = nm_res.x.copy()
        except Exception:
            pass
    
    # 3. Random restarts with local optimization
    if time.time() < _DEADLINE - 15:
        np.random.seed(123)
        for _ in range(8):
            if time.time() > _DEADLINE - 15:
                break
            # Random start near the best or in the bounds
            if np.random.rand() < 0.6:
                start = best_result + np.random.normal(0, 0.03, 3)
            else:
                start = np.array([
                    np.random.uniform(-0.5, 0.5),
                    np.random.uniform(-0.05, 0.05),
                    np.random.uniform(-0.005, 0.005)
                ])
            try:
                nm_res = minimize(
                    lambda x: _compute_bound(x, use_hp=False), 
                    start, 
                    method='Nelder-Mead', 
                    options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                if nm_res.fun < best_score:
                    best_score = nm_res.fun
                    best_result = nm_res.x.copy()
            except Exception:
                pass
    
    # 4. Final high-precision polish
    if time.time() < _DEADLINE - 5:
        try:
            nm_res_hp = minimize(
                lambda x: _compute_bound(x, use_hp=True), 
                best_result, 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if nm_res_hp.fun < best_score:
                best_score = nm_res_hp.fun
                best_result = nm_res_hp.x.copy()
        except Exception:
            pass

    return np.array(best_result)

# EVOLVE-BLOCK-END
