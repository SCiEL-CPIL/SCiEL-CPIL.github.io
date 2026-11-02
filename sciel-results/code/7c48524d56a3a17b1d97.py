"""Improved program: Exact rational function minimization for C4 bound."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_DEADLINE = 195  # safety margin before 210s kill

# Precompute weights w_n = 2^n n!
# We work with ratios, so sqrt(pi) cancels out.
w0 = 1.0
w4 = 2**4 * 24.0       # 4! = 24
w8 = 2**8 * 40320.0    # 8! = 40320
w12 = 2**12 * 479001600.0 # 12! = 479001600

# Coefficients for the constraint c12 = -(c0 + 12 c4 + 40320 c8) / H12(0)
# H12(0) = 103219200
H12_0 = 103219200.0

def _objective(params):
    """
    Minimize log(Qx) + log(Qk) - 2 log(N)
    which is equivalent to minimizing Qx * Qk / N^2
    """
    if time.time() > _DEADLINE:
        return float('inf')
    
    c0, c4, c8 = params
    
    # Calculate c12 from constraint g(0)=0
    c12 = -(c0 + 12.0 * c4 + 40320.0 * c8) / H12_0
    
    # Calculate u_n = c_n^2 w_n
    u0 = c0**2 * w0
    u4 = c4**2 * w4
    u8 = c8**2 * w8
    u12 = c12**2 * w12
    
    N = u0 + u4 + u8 + u12
    if N <= 1e-15:
        return float('inf')
        
    Qx = 1.0 * u0 + 9.0 * u4 + 17.0 * u8 + 25.0 * u12
    Qk = 0.0 * u0 + 8.0 * u4 + 16.0 * u8 + 24.0 * u12
    
    if Qx <= 1e-15 or Qk <= 1e-15:
        return float('inf')
        
    # We want to minimize log(Qx) + log(Qk) - 2*log(N)
    # Using log to avoid overflow/underflow issues, though ratios are safe
    val = np.log(Qx) + np.log(Qk) - 2.0 * np.log(N)
    return val

def _objective_grad(params):
    """
    Gradient of the objective function.
    """
    if time.time() > _DEADLINE:
        return float('inf'), np.zeros(3)
        
    c0, c4, c8 = params
    
    c12 = -(c0 + 12.0 * c4 + 40320.0 * c8) / H12_0
    
    u0 = c0**2 * w0
    u4 = c4**2 * w4
    u8 = c8**2 * w8
    u12 = c12**2 * w12
    
    N = u0 + u4 + u8 + u12
    if N <= 1e-15:
        return float('inf'), np.zeros(3)
        
    Qx = 1.0 * u0 + 9.0 * u4 + 17.0 * u8 + 25.0 * u12
    Qk = 0.0 * u0 + 8.0 * u4 + 16.0 * u8 + 24.0 * u12
    
    if Qx <= 1e-15 or Qk <= 1e-15:
        return float('inf'), np.zeros(3)
        
    # Derivatives of u_n with respect to c_i
    # du0/dc0 = 2 c0 w0
    # du4/dc4 = 2 c4 w4
    # du8/dc8 = 2 c8 w8
    # du12/dc0 = 2 c12 w12 * (-1/H12_0)
    # du12/dc4 = 2 c12 w12 * (-12/H12_0)
    # du12/dc8 = 2 c12 w12 * (-40320/H12_0)
    
    du0_dc0 = 2.0 * c0 * w0
    du4_dc4 = 2.0 * c4 * w4
    du8_dc8 = 2.0 * c8 * w8
    
    factor12 = 2.0 * c12 * w12 / H12_0
    du12_dc0 = -factor12
    du12_dc4 = -12.0 * factor12
    du12_dc8 = -40320.0 * factor12
    
    # dN/dc_i = du_i/dc_i
    dN_dc0 = du0_dc0 + du12_dc0
    dN_dc4 = du4_dc4 + du12_dc4
    dN_dc8 = du8_dc8 + du12_dc8
    
    # dQx/dc_i
    dQx_dc0 = 1.0 * du0_dc0 + 25.0 * du12_dc0
    dQx_dc4 = 9.0 * du4_dc4 + 25.0 * du12_dc4
    dQx_dc8 = 17.0 * du8_dc8 + 25.0 * du12_dc8
    
    # dQk/dc_i
    dQk_dc0 = 24.0 * du12_dc0
    dQk_dc4 = 8.0 * du4_dc4 + 24.0 * du12_dc4
    dQk_dc8 = 16.0 * du8_dc8 + 24.0 * du12_dc8
    
    # Objective: L = log(Qx) + log(Qk) - 2 log(N)
    # dL/dc_i = dQx/Qx + dQk/Qk - 2 dN/N
    
    grad0 = dQx_dc0 / Qx + dQk_dc0 / Qk - 2.0 * dN_dc0 / N
    grad4 = dQx_dc4 / Qx + dQk_dc4 / Qk - 2.0 * dN_dc4 / N
    grad8 = dQx_dc8 / Qx + dQk_dc8 / Qk - 2.0 * dN_dc8 / N
    
    val = np.log(Qx) + np.log(Qk) - 2.0 * np.log(N)
    
    return val, np.array([grad0, grad4, grad8])

def construct():
    start_time = time.time()
    
    # Seed points: 
    # 1. The original seed from the prompt
    # 2. Points near the theoretical minimum (often balanced coefficients)
    # 3. Random points on a sphere
    
    best_result = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = _objective(best_result)
    
    def _try_update(cand, val):
        nonlocal best_result, best_val
        if val < best_val:
            best_val = val
            best_result = cand.copy()
            
    # List of starting points
    starts = [
        np.array([-1.13, 0.04, 0.000308642]), # Seed
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([-1.0, 1.0, -1.0]),
        np.array([1.0, -1.0, 1.0]),
        np.array([-1.0, -1.0, -1.0]),
    ]
    
    # Add some random points
    np.random.seed(42)
    for _ in range(20):
        v = np.random.randn(3)
        v = v / np.linalg.norm(v)
        # Scale randomly to explore different magnitudes, though objective is scale invariant
        # Actually, scale invariance means only direction matters.
        # But c12 depends on absolute values. Wait.
        # c12 = -(c0 + 12c4 + 40320c8)/H12.
        # If we scale c by k, c12 scales by k.
        # u_n scales by k^2.
        # N, Qx, Qk all scale by k^2.
        # Ratio Qx Qk / N^2 is invariant to scaling.
        # So we can just search on the unit sphere.
        starts.append(v)
        
    for s in starts:
        if time.time() > _DEADLINE - 10:
            break
        try:
            res = minimize(_objective_grad, s, method='L-BFGS-B', jac=True,
                           options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-14})
            _try_update(res.x, res.fun)
        except Exception:
            pass
            
    # If we have time, do a more thorough random search
    if time.time() < _DEADLINE - 20:
        for _ in range(50):
            if time.time() > _DEADLINE - 20:
                break
            v = np.random.randn(3)
            v = v / np.linalg.norm(v)
            try:
                res = minimize(_objective_grad, v, method='L-BFGS-B', jac=True,
                               options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-13})
                _try_update(res.x, res.fun)
            except Exception:
                pass
                
    # Final polish
    if time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_result, method='L-BFGS-B', jac=True,
                           options={'maxiter': 20000, 'ftol': 1e-16, 'gtol': 1e-15})
            _try_update(res.x, res.fun)
        except Exception:
            pass
            
    return best_result

# EVOLVE-BLOCK-END
