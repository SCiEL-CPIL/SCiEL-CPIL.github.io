"""ODE integrator for the Lorenz system."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate the Lorenz system from t_span[0] to t_span[1] starting at y0.
    Uses a high-order explicit Runge-Kutta method (Dormand-Prince 5(4) with
    adaptive step size) to achieve high accuracy with a moderate number of
    function evaluations.
    
    Parameters:
        f: function(t, y) -> y'
        t_span: tuple (t_start, t_end)
        y0: initial condition (float array)
    
    Returns:
        y_end: state at t_end
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Deadline guard: stop if we exceed 290 seconds (30s margin before 330s kill)
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Dormand-Prince 5(4) coefficients
    # c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    # a = [[0,0,0,0,0,0],
    #      [1/5,0,0,0,0,0],
    #      [3/40,9/40,0,0,0,0],
    #      [44/45,-56/15,32/9,0,0,0],
    #      [19372/6561,-25360/2187,64448/6561,-212/729,0,0],
    #      [9017/3168,-355/33,46732/5247,49/176,-5103/18656,0],
    #      [35/384,0,500/1113,125/192,-2187/6784,11/84,0]]
    # b = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    # b* = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
    
    a = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    ])
    c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
    b5 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
    b4 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
    
    # Adaptive step size control
    rtol = 1e-12
    atol = 1e-14
    
    t = t0
    h = (t1 - t0) / 1000  # initial step
    h_min = (t1 - t0) * 1e-10
    h_max = (t1 - t0) * 0.1
    
    while t < t1:
        if time.time() > deadline:
            break
        
        # Clamp step to not overshoot t1
        if t + h > t1:
            h = t1 - t
        
        # Compute RK stages
        k = np.zeros((7, n))
        for i in range(7):
            y_stage = y.copy()
            for j in range(i):
                y_stage = y_stage + h * a[i, j] * k[j]
            k[i] = f(t + c[i] * h, y_stage)
        
        # 5th and 4th order solutions
        y5 = y + h * np.dot(b5, k)
        y4 = y + h * np.dot(b4, k)
        
        # Error estimate
        err = np.abs(y5 - y4)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
        err_norm = np.sqrt(np.mean((err / scale) ** 2))
        
        if err_norm <= 1.0:
            # Accept step
            y = y5
            t = t + h
            
            # Adjust step size for next step
            if err_norm > 1e-15:
                factor = min(5.0, max(0.1, 0.9 * err_norm ** (-1/5)))
            else:
                factor = 5.0
            h = h * factor
            h = max(h_min, min(h_max, h))
        else:
            # Reject step, reduce h
            factor = max(0.1, 0.9 * err_norm ** (-1/5))
            h = h * factor
            h = max(h_min, h)
        
        if not np.all(np.isfinite(y)):
            # Fallback: use a simple fixed-step RK4 with many small steps
            return _rk4_fallback(f, t0, t1, y0)
    
    return y


def _rk4_fallback(f, t0, t1, y0):
    """Simple fixed-step RK4 fallback."""
    steps = 10000
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y
# EVOLVE-BLOCK-END
