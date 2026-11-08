"""Improved ODE integrator using adaptive high-order Runge-Kutta for Lotka-Volterra."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Integrates the Lotka-Volterra system using an adaptive Dormand-Prince 5(4) method.
    
    Args:
    f: Function (t, y) -> y'
    t_span: Tuple (t_start, t_end)
    y0: Initial state vector
    
    Returns:
    y_end: Final state vector
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Dormand-Prince 5(4) coefficients
    # c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    # A matrix
    # b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    # b4 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
    
    c = [0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0]
    a = [
        [0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
    ]
    b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    b4 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
    
    # Initial step size estimation
    # Heuristic based on initial derivative
    y0_norm = np.linalg.norm(y)
    if y0_norm == 0:
        y0_norm = 1.0
        
    k0 = f(t0, y)
    k0_norm = np.linalg.norm(k0)
    if k0_norm == 0:
        k0_norm = 1e-6
        
    # Safety factor and error tolerances
    rtol = 1e-11
    atol = 1e-13
    safety = 0.9
    min_factor = 0.1
    max_factor = 10.0
    
    # Initial h
    h = 0.01 * (t1 - t0)
    if h == 0:
        h = 1e-6
    # Adjust h based on initial slope
    # h ~ (tol / slope)^(1/(order+1))
    h_init = (rtol * y0_norm / k0_norm) ** (1.0/5.0)
    h = min(abs(h), abs(h_init)) if h_init > 0 else abs(h)
    if h < 1e-12:
        h = 1e-12
        
    t = t0
    direction = 1.0
    if t1 < t0:
        direction = -1.0
        h = -abs(h)
    else:
        h = abs(h)
        
    deadline = time.time() + 290.0 # Leave 10s margin for 300s limit
    
    while (direction > 0 and t < t1) or (direction < 0 and t > t1):
        if time.time() > deadline:
            # Return current best estimate if time runs out
            return y
            
        # Don't overshoot t1
        if direction > 0 and t + h > t1:
            h = t1 - t
        elif direction < 0 and t + h < t1:
            h = t1 - t
            
        if abs(h) < 1e-15:
            break
            
        # Compute stages
        k = [None] * 7
        k[0] = f(t, y)
        
        for i in range(1, 7):
            y_stage = y + h * sum(a[i-1][j] * k[j] for j in range(i))
            k[i] = f(t + c[i] * h, y_stage)
            
        # Compute 5th and 4th order solutions
        y5 = y + h * sum(b5[i] * k[i] for i in range(7))
        y4 = y + h * sum(b4[i] * k[i] for i in range(7))
        
        # Error estimate
        err = np.linalg.norm(y5 - y4)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
        err_scaled = np.linalg.norm((y5 - y4) / scale)
        
        if err_scaled <= 1.0:
            # Step accepted
            t = t + h
            y = y5
            
            # Adjust step size
            if err_scaled < 1e-10:
                factor = max_factor
            else:
                factor = safety * (err_scaled ** (-0.2))
            factor = min(max_factor, max(min_factor, factor))
            h = h * factor
            
            # Clamp h to not overshoot significantly
            if direction > 0 and t + h > t1:
                h = t1 - t
            elif direction < 0 and t + h < t1:
                h = t1 - t
        else:
            # Step rejected
            factor = max(min_factor, safety * (err_scaled ** (-0.2)))
            h = h * factor
            
            # Prevent infinite loop if h becomes too small
            if abs(h) < 1e-15:
                break
                
    return y
# EVOLVE-BLOCK-END
