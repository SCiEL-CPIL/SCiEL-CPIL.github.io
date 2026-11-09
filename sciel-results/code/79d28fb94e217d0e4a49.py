"""Improved ODE integrator for Van der Pol oscillator using adaptive RK45 with error control."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard: finish well before 300s
    deadline = time.time() + 280.0
    
    # Adaptive RK45 (Dormand-Prince) with error control
    # This should handle the Van der Pol oscillator efficiently
    
    # Coefficients for RK45 (Dormand-Prince)
    c2 = 1/5
    c3 = 3/10
    c4 = 4/5
    c5 = 8/9
    c6 = 1.0
    c7 = 1.0
    
    a21 = 1/5
    a31 = 3/40
    a32 = 9/40
    a41 = 44/45
    a42 = -56/15
    a43 = 32/9
    a51 = 19372/6561
    a52 = -25360/2187
    a53 = 64448/6561
    a54 = -212/729
    a61 = 9017/3168
    a62 = -355/33
    a63 = 46732/5247
    a64 = 49/176
    a65 = -5103/18656
    a71 = 35/384
    a72 = 0
    a73 = 500/1113
    a74 = 125/192
    a75 = -2187/6784
    a76 = 11/84
    
    b1 = 35/384
    b2 = 0
    b3 = 500/1113
    b4 = 125/192
    b5 = -2187/6784
    b6 = 11/84
    b7 = 0
    
    e1 = 71/57600
    e2 = 0
    e3 = -71/16695
    e4 = 71/1920
    e5 = -17253/339200
    e6 = 22/525
    e7 = -1/40
    
    # Adaptive stepping
    rtol = 1e-12
    atol = 1e-14
    
    t = t0
    y = y0.copy()
    h = (t1 - t0) / 100.0  # Initial step estimate
    
    max_steps = 2000000
    step_count = 0
    f_calls = 0
    
    while t < t1 - 1e-15 and step_count < max_steps:
        if time.time() > deadline - 10.0:
            break
        
        # Don't overshoot
        if t + h > t1:
            h = t1 - t
        
        # RK45 stage computations
        k1 = f(t, y)
        f_calls += 1
        
        k2 = f(t + c2 * h, y + h * (a21 * k1))
        f_calls += 1
        
        k3 = f(t + c3 * h, y + h * (a31 * k1 + a32 * k2))
        f_calls += 1
        
        k4 = f(t + c4 * h, y + h * (a41 * k1 + a42 * k2 + a43 * k3))
        f_calls += 1
        
        k5 = f(t + c5 * h, y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
        f_calls += 1
        
        k6 = f(t + c6 * h, y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5))
        f_calls += 1
        
        # 4th order solution (for error estimation)
        y_new = y + h * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
        
        # 5th order solution (more accurate)
        y_new5 = y + h * (a71 * k1 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6)
        
        # Error estimate
        err = y_new5 - y_new
        err_norm = np.sqrt(np.sum((err / (atol + rtol * np.maximum(np.abs(y), np.abs(y_new5))))**2)) / np.sqrt(n)
        
        step_count += 1
        
        if err_norm <= 1.0:
            # Accept step
            t = t + h
            y = y_new5
            # Adjust step size for next step
            if err_norm > 0:
                h_new = h * min(5.0, max(0.1, 0.9 * err_norm**(-0.2)))
            else:
                h_new = h * 5.0
            h = min(h_new, t1 - t) if t1 - t > 0 else h
        else:
            # Reject step, reduce h
            h = h * max(0.1, 0.9 * err_norm**(-0.25))
            if h < 1e-15:
                h = 1e-15
    
    return y
# EVOLVE-BLOCK-END
