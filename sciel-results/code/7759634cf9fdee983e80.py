import numpy as np
from scipy.optimize import minimize
import time

def construct():
    # Hermite polynomial values at 0 (physicists' convention)
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    # Normalization constants (without sqrt(pi) factor, which cancels)
    N0 = 1.0
    N4 = 384.0
    N8 = 10321920.0
    N12 = 1962007275520.0
    
    # Coefficients for <x^2> numerator: (n + 0.5) * N_n
    C0_num = 0.5 * N0
    C4_num = 4.5 * N4
    C8_num = 8.5 * N8
    C12_num = 12.5 * N12
    
    # c12 is determined by g(0) = 0:
    # c0 * H0_0 + c4 * H4_0 + c8 * H8_0 + c12 * H12_0 = 0
    # c12 = -(c0 + 12*c4 + 1680*c8) / 665280
    
    def c12_from(c0, c4, c8):
        return -(c0 + H4_0 * c4 + H8_0 * c8) / H12_0
    
    def objective(c):
        c0, c4, c8 = c
        c12 = c12_from(c0, c4, c8)
        
        num = C0_num * c0**2 + C4_num * c4**2 + C8_num * c8**2 + C12_num * c12**2
        den = N0 * c0**2 + N4 * c4**2 + N8 * c8**2 + N12 * c12**2
        
        if den <= 0:
            return np.inf
        return num / den
    
    def gradient(c):
        c0, c4, c8 = c
        c12 = c12_from(c0, c4, c8)
        
        dc12_dc0 = -1.0 / H12_0
        dc12_dc4 = -H4_0 / H12_0
        dc12_dc8 = -H8_0 / H12_0
        
        num = C0_num * c0**2 + C4_num * c4**2 + C8_num * c8**2 + C12_num * c12**2
        den = N0 * c0**2 + N4 * c4**2 + N8 * c8**2 + N12 * c12**2
        
        if den <= 0:
            return np.array([0.0, 0.0, 0.0])
        
        dnum_dc0 = 2 * C0_num * c0 + 2 * C12_num * c12 * dc12_dc0
        dnum_dc4 = 2 * C4_num * c4 + 2 * C12_num * c12 * dc12_dc4
        dnum_dc8 = 2 * C8_num * c8 + 2 * C12_num * c12 * dc12_dc8
        
        dden_dc0 = 2 * N0 * c0 + 2 * N12 * c12 * dc12_dc0
        dden_dc4 = 2 * N4 * c4 + 2 * N12 * c12 * dc12_dc4
        dden_dc8 = 2 * N8 * c8 + 2 * N12 * c12 * dc12_dc8
        
        grad0 = (dnum_dc0 * den - num * dden_dc0) / (den**2)
        grad4 = (dnum_dc4 * den - num * dden_dc4) / (den**2)
        grad8 = (dnum_dc8 * den - num * dden_dc8) / (den**2)
        
        return np.array([grad0, grad4, grad8])
    
    # Deadline guard
    deadline = time.time() + 200  # 10s safety margin before 210s kill
    
    # Multi-start optimization with diverse initial points
    # The problem is a Rayleigh quotient with a linear constraint, so it's convex-ish
    # but the constraint creates a non-trivial landscape
    
    # Good initial points based on analysis
    starts = [
        # Points that satisfy g(0)=0 approximately
        np.array([-1.0, 1.0/12.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([1.0, 0.0, -1.0/1680.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([1.0, -1.0/12.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([-0.5, 0.1, 0.0001]),
        np.array([0.5, -0.1, -0.0001]),
        np.array([-2.0, 0.3, 0.0005]),
        np.array([2.0, -0.3, -0.0005]),
        np.array([-1.0, 0.5, 0.001]),
        np.array([1.0, -0.5, -0.001]),
        np.array([-3.0, 1.0, 0.002]),
        np.array([3.0, -1.0, -0.002]),
    ]
    
    best_val = np.inf
    best_x = None
    
    for x0 in starts:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 50})
            val = objective(res.x)
            if val < best_val:
                best_val = val
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Additional random starts if time permits
    if time.time() < deadline - 20:
        rng = np.random.default_rng(12345)
        for _ in range(30):
            if time.time() > deadline:
                break
            # Random starts with varying scales
            scale = 10**rng.uniform(-2, 2)
            x0 = rng.normal(0, scale, 3)
            try:
                res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                              options={'maxiter': 300, 'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 50})
                val = objective(res.x)
                if val < best_val:
                    best_val = val
                    best_x = res.x.copy()
            except Exception:
                pass
    
    # Fine-tune with tighter tolerances if we have a good starting point
    if best_x is not None and time.time() < deadline - 10:
        try:
            res = minimize(objective, best_x, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': 1000, 'ftol': 1e-20, 'gtol': 1e-16, 'maxls': 100})
            val = objective(res.x)
            if val < best_val:
                best_val = val
                best_x = res.x.copy()
        except Exception:
            pass
    
    if best_x is None:
        # Fallback: return a reasonable default
        return np.array([-1.0, 1.0/12.0, 0.0])
    
    return best_x
