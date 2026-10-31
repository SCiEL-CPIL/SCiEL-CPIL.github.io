import numpy as np
import time
import scipy.optimize as opt
import scipy.special as sp
from scipy.integrate import quad

# Time guard
START_TIME = time.time()
TIME_LIMIT = 170  # Safety margin for 180s limit

def _time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def _hermite(n, x):
    # Physicists' Hermite polynomials
    # H_0 = 1
    # H_1 = 2x
    # H_{n+1} = 2x H_n - 2n H_{n-1}
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    H_prev = np.ones_like(x)
    H_curr = 2 * x
    for k in range(1, n):
        H_next = 2 * x * H_curr - 2 * k * H_prev
        H_prev = H_curr
        H_curr = H_next
    return H_curr

def _compute_c12(c0, c4, c8):
    # P(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    # H0(0) = 1
    # H4(0) = -12
    # H8(0) = 1008
    # H12(0) = -145152
    return (c0 * 1 + c4 * (-12) + c8 * 1008) / 145152

def _objective(coeffs):
    c0, c4, c8 = coeffs
    c12 = _compute_c12(c0, c4, c8)
    
    # Define P(x)
    def P(x):
        return c0 * _hermite(0, x) + c4 * _hermite(4, x) + c8 * _hermite(8, x) + c12 * _hermite(12, x)
    
    # Integrand for L1 norm
    def integrand(x):
        return np.abs(np.exp(-x**2 / 2.0) * P(x))
    
    # Integrate from -infinity to infinity
    # Use symmetry: 2 * integral from 0 to infinity
    val, err = quad(integrand, 0, np.inf, limit=200)
    return 2 * val

def construct():
    best_coeffs = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_val = _objective(best_coeffs)
    
    # Local optimization starting from seed
    try:
        res = opt.minimize(_objective, best_coeffs, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.success and res.fun < best_val:
            best_coeffs = res.x
            best_val = res.fun
    except Exception:
        pass
    
    if _time_left() < 20:
        return best_coeffs

    # Try multiple random restarts if time permits
    np.random.seed(42)
    for _ in range(5):
        if _time_left() < 20:
            break
        # Random perturbation around seed
        noise = np.random.normal(0, 0.5, 3)
        start = best_coeffs + noise
        try:
            res = opt.minimize(_objective, start, method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.success and res.fun < best_val:
                best_coeffs = res.x
                best_val = res.fun
        except Exception:
            pass
            
    return best_coeffs
