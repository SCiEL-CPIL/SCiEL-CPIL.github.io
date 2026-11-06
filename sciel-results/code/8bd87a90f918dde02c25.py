"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Solve the ODE y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Uses an adaptive Runge-Kutta 4(5) method (Dormand-Prince) with a deadline guard.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Deadline: leave 30s margin before the 330s limit
    deadline = time.time() + 300.0
    
    # Dormand-Prince coefficients
    a2, a3, a4, a5, a6, a7 = 1/5, 3/10, 4/5, 8/9, 1, 1
    b2 = 1/5
    b3 = 3/40, 9/40
    b4 = 44/45, -56/15, 32/9
    b5 = 19372/6561, -25360/2187, 64448/6561, -212/729
    b6 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
    b7 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84
    
    # Error estimation coefficients (5th order)
    e = np.array([71/57600, 0, -71/16695, 71/1920, -17253/339200, 22/525, -1/40])
    
    # 4th order solution coefficients
    c4 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
    
    # Adaptive step control
    rtol = 1e-8
    atol = 1e-10
    h = (t1 - t0) / 1000  # initial step
    h_min = (t1 - t0) / 1e6
    h_max = (t1 - t0) / 10
    safety = 0.9
    min_factor = 0.2
    max_factor = 10.0
    
    t = t0
    call_count = 0
    
    while t < t1 - 1e-14:
        if time.time() > deadline:
            break
        
        # Don't overshoot t1
        if t + h > t1:
            h = t1 - t
        
        # Dormand-Prince step
        k1 = f(t, y)
        call_count += 1
        
        k2 = f(t + a2 * h, y + h * (b2 * k1))
        call_count += 1
        
        k3 = f(t + a3 * h, y + h * (b3[0] * k1 + b3[1] * k2))
        call_count += 1
        
        k4 = f(t + a4 * h, y + h * (b4[0] * k1 + b4[1] * k2 + b4[2] * k3))
        call_count += 1
        
        k5 = f(t + a5 * h, y + h * (b5[0] * k1 + b5[1] * k2 + b5[2] * k3 + b5[3] * k4))
        call_count += 1
        
        k6 = f(t + a6 * h, y + h * (b6[0] * k1 + b6[1] * k2 + b6[2] * k3 + b6[3] * k4 + b6[4] * k5))
        call_count += 1
        
        k7 = f(t + a7 * h, y + h * (b7[0] * k1 + b7[1] * k2 + b7[2] * k3 + b7[3] * k4 + b7[4] * k5 + b7[5] * k6))
        call_count += 1
        
        # 5th order solution
        y5 = y + h * (b7[0] * k1 + b7[1] * k2 + b7[2] * k3 + b7[3] * k4 + b7[4] * k5 + b7[5] * k6)
        
        # Error estimate
        err = h * (e[0] * k1 + e[1] * k2 + e[2] * k3 + e[3] * k4 + e[4] * k5 + e[5] * k6 + e[6] * k7)
        
        # Scale for error control
        scale = atol + rtol * np.abs(y)
        err_norm = np.max(np.abs(err) / scale)
        
        if err_norm <= 1.0:
            # Accept step
            t += h
            y = y5
            # Increase step size
            if err_norm > 1e-15:
                factor = safety * err_norm ** (-0.2)
            else:
                factor = max_factor
            h = min(h * max(min_factor, min(max_factor, factor)), h_max, t1 - t)
        else:
            # Reject step, decrease h
            factor = safety * err_norm ** (-0.25)
            h = max(h * max(min_factor, min(0.5, factor)), h_min)
    
    return y
# EVOLVE-BLOCK-END
