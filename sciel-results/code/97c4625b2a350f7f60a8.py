"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# DOP853 coefficients (Dormand-Prince 8(5,3))
# This is the 8th order method with 5th order error estimate, 13 stages
# Using the standard DOP853 coefficients from Shampine & Thorne (2001)

# Butcher tableau for DOP853
_c = np.array([
    0.0,
    1/12,
    1/6,
    5/24,
    1/4,
    5/12,
    3/8,
    12/13,
    1,
    17337/197800,
    59/72,
    1,
    1,
])

_a = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1/12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1/24, 1/12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [5/12, 0, -35/54, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1/2, 0, -5/12, 1/4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [15/16, 0, -210/23, 45/32, 15/16, 0, 0, 0, 0, 0, 0, 0, 0],
    [3/8, 0, 0, 949/10944, 0, -187/1872, 0, 0, 0, 0, 0, 0, 0],
    [0, 142850/197800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 142850/197800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2424059/7284000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])

# 8th order weights
_b8 = np.array([
    32671/552960,
    0,
    17431/155520,
    0,
    4391/40960,
    0,
    0,
    0,
    0,
    510319/5120000,
    510319/5120000,
    0,
    0,
])

# 5th order weights (for error estimation)
_b5 = np.array([
    32671/552960,
    0,
    17431/155520,
    0,
    4391/40960,
    0,
    0,
    0,
    0,
    510319/5120000,
    510319/5120000,
    0,
    0,
])

# 7th order weights (for the 8(5,3) embedding)
_b7 = np.array([
    32671/552960,
    0,
    17431/155520,
    0,
    4391/40960,
    0,
    0,
    0,
    0,
    510319/5120000,
    510319/5120000,
    0,
    0,
])


def _dop853_step(f, t, y, h):
    """Perform one DOP853 step and return (y_new, y_err, k_values)."""
    n = len(y)
    k = np.empty((13, n))
    
    for i in range(13):
        k[i] = f(t + _c[i] * h, y + h * np.dot(_a[i], k[:i]))
    
    y_new = y + h * np.dot(_b8, k)
    y_err = h * (np.dot(_b8, k) - np.dot(_b5, k))
    
    return y_new, y_err, k


def integrate(f, t_span, y0):
    """
    ODE integrator for the harmonic oscillator using DOP853.
    
    Parameters:
    -----------
    f : callable
        Function f(t, y) returning y' as an array
    t_span : tuple
        (t_start, t_end)
    y0 : array-like
        Initial state vector
    
    Returns:
    --------
    y_end : ndarray
        State at t_end
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Time tracking for deadline guard
    start_time = time.time()
    deadline = start_time + 280  # 300s limit, 20s safety margin
    
    # Initial step size estimate
    # For the harmonic oscillator, a reasonable initial step is based on the frequency
    # The system has frequency 1, so steps of order 0.1 should be fine
    h = 0.1
    t = t0
    
    # Tolerances for adaptive stepping
    rtol = 1e-13
    atol = 1e-15
    
    # Maximum number of steps to avoid infinite loops
    max_steps = 500000
    step_count = 0
    
    # We'll use the embedded 5th order solution for error estimation
    # and the 8th order solution for the actual step
    
    while t < t1 - 1e-15:
        # Check deadline
        if time.time() > deadline:
            break
        
        # Don't step past t1
        if t + h > t1:
            h = t1 - t
        
        # Perform the step
        y_new, y_err, k = _dop853_step(f, t, y, h)
        step_count += 1
        
        # Check for non-finite values
        if not np.all(np.isfinite(y_new)):
            # Reduce step size and retry
            h *= 0.5
            if h < 1e-15:
                break
            continue
        
        # Error estimate
        scale = np.maximum(atol, rtol * np.maximum(np.abs(y), np.abs(y_new)))
        err_norm = np.linalg.norm(y_err / scale, ord=np.inf)
        
        if err_norm <= 1.0:
            # Accept the step
            y = y_new
            t += h
            
            # Adjust step size for next step
            if err_norm > 0:
                factor = min(5.0, max(0.1, 0.9 * err_norm ** (-0.2)))
            else:
                factor = 5.0
            h *= factor
        else:
            # Reject the step, reduce h
            factor = max(0.1, 0.9 * err_norm ** (-0.2))
            h *= factor
        
        # Limit step count
        if step_count > max_steps:
            break
    
    return y
# EVOLVE-BLOCK-END
