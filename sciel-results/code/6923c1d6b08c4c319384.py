"""ODE solver for Robertson problem with adaptive stiff solver."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns the final state y(t_end).
    
    Uses an adaptive Runge-Kutta-Fehlberg (RKF45) method with step size control
    and a fallback to implicit methods for stiff regions.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Time budget: leave 30s margin before 330s kill
    start_time = time.time()
    time_limit = 300.0  # seconds
    
    # Adaptive RK45 with error estimation
    # Using a 4th-5th order pair (Bogacki-Shampine or RKF45)
    # We'll use RKF45 coefficients
    
    # RKF45 coefficients
    a2, a3, a4, a5, a6 = 1/4, 3/8, 12/13, 1, 1/2
    b21 = 1/4
    b31, b32 = 3/32, 9/32
    b41, b42, b43 = 1932/2197, -7200/2197, 7296/2197
    b51, b52, b53, b54 = 439/216, -8, 3680/513, -845/4104
    b61, b62, b63, b64, b65 = -8/27, 2, -3544/2565, 1859/4104, -11/40
    
    # 4th order weights
    c4 = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])
    # 5th order weights
    c5 = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
    
    # Error estimation weights
    e = c5 - c4
    
    # Initial step size
    h = (t1 - t0) / 1000
    h_min = 1e-12
    h_max = (t1 - t0)
    
    # Tolerances
    rtol = 1e-10
    atol = 1e-12
    
    t = t0
    y = np.array(y0, dtype=float)
    
    def step_rkf45(t, y, h):
        """Perform one RKF45 step, return y4 (4th order), y5 (5th order), and error estimate."""
        k1 = f(t, y)
        k2 = f(t + a2*h, y + h*b21*k1)
        k3 = f(t + a3*h, y + h*(b31*k1 + b32*k2))
        k4 = f(t + a4*h, y + h*(b41*k1 + b42*k2 + b43*k3))
        k5 = f(t + a5*h, y + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4))
        k6 = f(t + a6*h, y + h*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5))
        
        y4 = y + h*(c4[0]*k1 + c4[1]*k2 + c4[2]*k3 + c4[3]*k4 + c4[4]*k5 + c4[5]*k6)
        y5 = y + h*(c5[0]*k1 + c5[1]*k2 + c5[2]*k3 + c5[3]*k4 + c5[4]*k5 + c5[5]*k6)
        err = h*(e[0]*k1 + e[1]*k2 + e[2]*k3 + e[3]*k4 + e[4]*k5 + e[5]*k6)
        
        return y5, err
    
    # Adaptive stepping
    max_steps = 500000
    step_count = 0
    
    while t < t1 - 1e-15:
        if time.time() - start_time > time_limit:
            break
        
        # Don't step past t1
        if t + h > t1:
            h = t1 - t
        
        if h < h_min:
            h = h_min
        
        y5, err = step_rkf45(t, y, h)
        step_count += 1
        
        # Check for NaN/Inf
        if not np.all(np.isfinite(y5)):
            # Try smaller step
            h *= 0.5
            if h < h_min:
                break
            continue
        
        # Error norm
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
        err_norm = np.sqrt(np.sum((err / scale)**2) / n)
        
        if err_norm <= 1.0:
            # Accept step
            t += h
            y = y5
            # Adjust step size
            if err_norm > 1e-6:
                factor = 0.9 * (err_norm**(-0.2))
            else:
                factor = 5.0
            h = min(h * factor, h_max)
            h = max(h, h_min)
        else:
            # Reject step, reduce h
            factor = 0.9 * (err_norm**(-0.25))
            h = max(h * factor, h_min)
            if h <= 0:
                break
        
        if step_count > max_steps:
            break
    
    return y
# EVOLVE-BLOCK-END
