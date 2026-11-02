"""Rewritten program for the suite task; optimized for uncertainty inequality."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_hermite
import time
import math

_DEADLINE = 180  # safety margin before 210s kill

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
    elif n == 8:
        return 256*x**8 - 4480*x**6 + 26880*x**4 - 67200*x**2 + 40320
    elif n == 12:
        return (4096*x**12 - 147456*x**10 + 2580480*x**8 
                - 20321280*x**6 + 81729600*x**4 
                - 172972800*x**2 + 103219200)
    else:
        raise ValueError(f"Hermite H_{n} not implemented")

# Precompute quadrature data for fast evaluation
_quad_data = None

def _get_quad_data():
    global _quad_data
    if _quad_data is None:
        n_quad = 200  # Higher precision quadrature
        x_quad, w_quad = roots_hermite(n_quad)
        
        H0 = _hermite_h(0, x_quad)
        H4 = _hermite_h(4, x_quad)
        H8 = _hermite_h(8, x_quad)
        H12 = _hermite_h(12, x_quad)
        
        # Derivatives: H_n'(x) = 2n H_{n-1}(x)
        dH0 = np.zeros_like(x_quad)
        dH4 = 8 * _hermite_h(3, x_quad)
        dH8 = 16 * _hermite_h(7, x_quad)
        dH12 = 24 * _hermite_h(11, x_quad)
        
        _quad_data = (x_quad, w_quad, H0, H4, H8, H12, dH0, dH4, dH8, dH12)
    return _quad_data

def _compute_bound(coeffs):
    """
    Compute the upper bound on C4 (uncertainty product) given coefficients [a0, a4, a8].
    The H12 coefficient d is fixed by g(0) = 0.
    """
    if time.time() > _DEADLINE:
        return float('inf')
    
    a0, a4, a8 = coeffs
    
    # H_n(0) values
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 40320.0
    H12_0 = 103219200.0
    
    # Constraint: g(0) = 0
    d = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    x_quad, w_quad, H0, H4, H8, H12, dH0, dH4, dH8, dH12 = _get_quad_data()
    
    # Construct P(x) and P'(x)
    P = a0 * H0 + a4 * H4 + a8 * H8 + d * H12
    dP = a0 * dH0 + a4 * dH4 + a8 * dH8 + d * dH12
    
    P2 = P * P
    norm2 = np.sum(w_quad * P2)
    
    if norm2 <= 1e-15:
        return 1e10
        
    # sigma_x^2 = <x^2>
    var_x = np.sum(w_quad * x_quad**2 * P2) / norm2
    
    # sigma_k^2 = <k^2> = <g'^2> / <g^2>
    # g'(x) = (P'(x) - x P(x)) * exp(-x^2/2)
    deriv_term = dP - x_quad * P
    deriv_term_sq = deriv_term * deriv_term
    norm_deriv2 = np.sum(w_quad * deriv_term_sq)
    
    var_k = norm_deriv2 / norm2
    
    bound = var_x * var_k
    
    return bound

def construct():
    start_time = time.time()
    
    # Seed solution from current program
    seed = np.array([-1.13, 0.04, 1/3240.0])
    
    best_result = seed.copy()
    best_score = _compute_bound(seed)
    
    # 1. Local optimization from seed
    def objective(params):
        if time.time() > _DEADLINE:
            return float('inf')
        return _compute_bound(params)
    
    try:
        # Use BFGS for robust local convergence
        result = minimize(objective, seed, method='BFGS', 
                         options={'maxiter': 500, 'gtol': 1e-10, 'disp': False})
        if result.fun < best_score:
            best_score = result.fun
            best_result = result.x.copy()
    except Exception:
        pass

    # 2. Multi-start strategy
    # The optimal coefficients are likely in a bounded region.
    # Based on the seed, a0 ~ -1, a4 ~ 0.04, a8 ~ 0.0003.
    # We sample a grid around the seed and some random points.
    
    if time.time() < _DEADLINE - 30:
        # Grid search around the seed
        a0_range = np.linspace(-2.0, 0.0, 15)
        a4_range = np.linspace(-0.1, 0.2, 15)
        a8_range = np.linspace(-0.002, 0.005, 15)
        
        # To save time, we don't do a full 3D grid (3375 points * ~1ms each = 3s, which is fine)
        # But let's be smarter. We'll do a few random restarts and a coarse grid.
        
        # Coarse Grid
        for a0 in a0_range[::3]:
            for a4 in a4_range[::3]:
                for a8 in a8_range[::3]:
                    if time.time() > _DEADLINE - 30:
                        break
                    start_point = np.array([a0, a4, a8])
                    try:
                        res = minimize(objective, start_point, method='BFGS', 
                                     options={'maxiter': 200, 'gtol': 1e-8, 'disp': False})
                        if res.fun < best_score:
                            best_score = res.fun
                            best_result = res.x.copy()
                    except Exception:
                        pass
            
            if time.time() > _DEADLINE - 30:
                break
        
        # Random Restarts
        np.random.seed(42)
        for _ in range(50):
            if time.time() > _DEADLINE - 30:
                break
            # Sample in a log-uniform or uniform range that covers plausible values
            r_a0 = np.random.uniform(-3.0, 1.0)
            r_a4 = np.random.uniform(-0.5, 0.5)
            r_a8 = np.random.uniform(-0.01, 0.02)
            start_point = np.array([r_a0, r_a4, r_a8])
            
            try:
                res = minimize(objective, start_point, method='BFGS', 
                             options={'maxiter': 200, 'gtol': 1e-8, 'disp': False})
                if res.fun < best_score:
                    best_score = res.fun
                    best_result = res.x.copy()
            except Exception:
                pass

    # 3. Final High-Precision Polish
    if time.time() < _DEADLINE - 5:
        try:
            # Use Nelder-Mead with very tight tolerances for final polish
            result = minimize(objective, best_result, method='Nelder-Mead', 
                             options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12, 'disp': False})
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            pass
        
        # One more BFGS pass with tight tolerances
        try:
            result = minimize(objective, best_result, method='BFGS', 
                             options={'maxiter': 1000, 'gtol': 1e-14, 'disp': False})
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            pass

    return np.array(best_result)

# EVOLVE-BLOCK-END
