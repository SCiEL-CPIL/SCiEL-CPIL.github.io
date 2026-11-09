"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra IVP from t_span[0] to t_span[1] using a
    custom adaptive Dormand-Prince (DP5) integrator with tight error control.
    
    The DP5 method uses 6 stages and provides a 5th-order solution with a
    4th-order error estimate, allowing for adaptive step size control.
    We use very tight tolerances (rtol=1e-13, atol=1e-15) to drive the
    relative error below the 1e-14 floor.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Dormand-Prince coefficients
    # b: 5th order weights
    b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
    # bhat: 4th order weights (for error estimation)
    bhat = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
    # c: stage times
    c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
    # A: Butcher table
    A = np.array([
        [0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    ])
    
    rtol = 1e-13
    atol = 1e-15
    min_h = 1e-10
    max_h = (t1 - t0) / 10  # initial step estimate
    max_steps = 2000000  # safety limit
    
    t = t0
    h = max_h
    step_count = 0
    
    while t < t1 and step_count < max_steps:
        # Ensure we don't overshoot t1
        if t + h > t1:
            h = t1 - t
            if h < min_h:
                h = min_h
        
        # Compute stages
        k = np.zeros((6, n))
        for i in range(6):
            ti = t + c[i] * h
            yi = y + h * np.dot(A[i, :i+1], k[:i+1])
            k[i] = f(ti, yi)
        
        # 5th order solution
        y_new = y + h * np.dot(b[:6], k[:6])
        # 4th order solution (for error estimate)
        y_err = y + h * np.dot(bhat[:6], k[:6])
        
        # Error estimate
        err = np.abs(y_new - y_err)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_norm = np.max(err / scale)
        
        # Accept step if error is within tolerance
        if err_norm <= 1.0 or h <= min_h:
            t += h
            y = y_new
            step_count += 1
            
            # Adjust step size for next step
            if err_norm > 0:
                # Optimal step size factor
                factor = min(5.0, max(0.1, 0.9 * err_norm ** (-0.2)))
                h_new = h * factor
            else:
                h_new = h * 5.0
            
            # Clamp step size
            h_new = max(min_h, min(h_new, (t1 - t) if t1 - t > 0 else max_h))
            h = h_new
        else:
            # Reject step, reduce h
            if err_norm > 0:
                factor = max(0.1, 0.9 * err_norm ** (-0.2))
            else:
                factor = 0.1
            h = max(min_h, h * factor)
    
    return y
# EVOLVE-BLOCK-END
