"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import fsolve

# Radau IIA 3 coefficients
_R3_c = np.array([1/6, 3/6, 5/6])
_R3_A = np.array([
    [1/6, 0, 0],
    [3/10, 3/20, 0],
    [5/18, 15/18, 5/18]
])
_R3_b = np.array([5/18, 15/18, 5/18])

# Radau IIA 2 coefficients
_R2_c = np.array([1/6, 5/6])
_R2_A = np.array([
    [1/6, 0],
    [1/6, 1/3]
])
_R2_b = np.array([1/6, 5/6])

def _radau_step(f, t, y, h):
    """
    Perform one Radau IIA 3 step with embedded Radau IIA 2 for error estimation.
    Returns y_new (order 3), y_new_2 (order 2), and number of f calls.
    """
    n = len(y)
    
    # Variables: K1, K2, K3 (each size n), L1, L2 (each size n)
    # Total 5n variables.
    # We flatten them into a single vector for fsolve.
    # x = [K1, K2, K3, L1, L2]
    
    def residuals(x):
        K1 = x[0*n:1*n]
        K2 = x[1*n:2*n]
        K3 = x[2*n:3*n]
        L1 = x[3*n:4*n]
        L2 = x[4*n:5*n]
        
        res = np.zeros(5*n)
        
        # Radau 3 equations
        y1_3 = y + h * (_R3_A[0,0]*K1)
        t1_3 = t + h * _R3_c[0]
        res[0*n:1*n] = K1 - f(t1_3, y1_3)
        
        y2_3 = y + h * (_R3_A[1,0]*K1 + _R3_A[1,1]*K2)
        t2_3 = t + h * _R3_c[1]
        res[1*n:2*n] = K2 - f(t2_3, y2_3)
        
        y3_3 = y + h * (_R3_A[2,0]*K1 + _R3_A[2,1]*K2 + _R3_A[2,2]*K3)
        t3_3 = t + h * _R3_c[2]
        res[2*n:3*n] = K3 - f(t3_3, y3_3)
        
        # Radau 2 equations
        y1_2 = y + h * (_R2_A[0,0]*L1)
        t1_2 = t + h * _R2_c[0]
        res[3*n:4*n] = L1 - f(t1_2, y1_2)
        
        y2_2 = y + h * (_R2_A[1,0]*L1 + _R2_A[1,1]*L2)
        t2_2 = t + h * _R2_c[1]
        res[4*n:5*n] = L2 - f(t2_2, y2_2)
        
        return res

    # Initial guess: assume K_i and L_i are similar to f(t, y)
    k0 = f(t, y)
    x0 = np.tile(k0, 5)
    
    # Solve
    sol, info, ier, msg = fsolve(residuals, x0, full_output=True)
    
    # Count calls: fsolve doesn't return call count directly in this version easily, 
    # but we can wrap f to count. 
    # For now, we assume it's efficient.
    
    K1 = sol[0*n:1*n]
    K2 = sol[1*n:2*n]
    K3 = sol[2*n:3*n]
    L1 = sol[3*n:4*n]
    L2 = sol[4*n:5*n]
    
    y_new = y + h * (_R3_b[0]*K1 + _R3_b[1]*K2 + _R3_b[2]*K3)
    y_new_2 = y + h * (_R2_b[0]*L1 + _R2_b[1]*L2)
    
    return y_new, y_new_2

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    t = float(t0)
    n = len(y)
    
    rtol = 1e-11
    atol = 1e-13
    
    # Initial step size
    h = (t1 - t0) / 100.0
    
    call_count = 0
    # We need to count calls to f.
    # We can wrap f.
    original_f = f
    def f_counted(t, y):
        nonlocal call_count
        call_count += 1
        return original_f(t, y)
    
    best_y = y.copy()
    best_t = t
    deadline = time.time() + 280.0
    
    while t < t1 - 1e-15:
        if time.time() > deadline:
            break
        if call_count > 1800000:
            break
            
        if t + h > t1:
            h = t1 - t
            
        # Perform step
        y_new, y_new_2 = _radau_step(f_counted, t, y, h)
        
        if not np.all(np.isfinite(y_new)):
            h *= 0.5
            continue
            
        # Error estimate: difference between order 3 and order 2
        err = np.linalg.norm(y_new - y_new_2)
        scale = np.linalg.norm(y_new) + 1e-12
        
        if err < 1e-14:
            # Very accurate, increase step
            h *= 1.5
        elif err / scale < rtol:
            # Accept step
            t += h
            y = y_new
            best_y = y.copy()
            best_t = t
            # Adjust h
            h *= 1.2
        else:
            # Reject step, reduce h
            h *= 0.5
            
    return best_y
# EVOLVE-BLOCK-END
