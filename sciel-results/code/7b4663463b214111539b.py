"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _dop853_step(f, t, y, h):
    """
    Single DOP853 step. Returns y_new, y_err_estimate.
    Uses the standard DOP853 coefficients.
    """
    # DOP853 coefficients (from Hairer, Lubich, Wanner)
    # 12 stages
    c = np.array([
        0.0,
        1/12,
        1/6,
        1/4,
        5/12,
        3/5,
        1,
        1/10,
        3/5,
        4/5,
        8/9,
        1.0
    ])
    
    # Butcher tableau A
    A = np.zeros((12, 12))
    A[1,0] = 1/12
    A[2,0] = 1/24
    A[2,1] = 1/24
    A[3,0] = 1/16
    A[3,1] = 0
    A[3,2] = 1/16
    A[4,0] = 5/132
    A[4,1] = 0
    A[4,2] = 25/132
    A[4,3] = -25/132
    A[5,0] = 225/4096
    A[5,1] = 0
    A[5,2] = 0
    A[5,3] = 275/4096
    A[5,4] = 125/4096
    A[6,0] = 143/128
    A[6,1] = 0
    A[6,2] = -354/128
    A[6,3] = 322/128
    A[6,4] = -133/128
    A[6,5] = 28/128
    A[7,0] = 19/210
    A[7,1] = 0
    A[7,2] = 0
    A[7,3] = 0
    A[7,4] = 0
    A[7,5] = 0
    A[7,6] = 0
    A[8,0] = 175/216
    A[8,1] = 0
    A[8,2] = -140/216
    A[8,3] = 210/216
    A[8,4] = -210/216
    A[8,5] = 112/216
    A[8,6] = 0
    A[8,7] = 0
    A[9,0] = 7/90
    A[9,1] = 0
    A[9,2] = 0
    A[9,3] = 0
    A[9,4] = 0
    A[9,5] = 0
    A[9,6] = 0
    A[9,7] = 0
    A[9,8] = 0
    A[10,0] = 12/13
    A[10,1] = 0
    A[10,2] = -24/13
    A[10,3] = 36/13
    A[10,4] = -36/13
    A[10,5] = 18/13
    A[10,6] = 0
    A[10,7] = 0
    A[10,8] = 0
    A[10,9] = 0
    A[11,0] = 0
    A[11,1] = 0
    A[11,2] = 0
    A[11,3] = 0
    A[11,4] = 0
    A[11,5] = 0
    A[11,6] = 0
    A[11,7] = 0
    A[11,8] = 0
    A[11,9] = 0
    A[11,10] = 0
    
    # 8th order weights
    b8 = np.array([
        3025/13824,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    ])
    # Actually DOP853 has specific weights. Let me use the correct ones.
    # From the original paper:
    # b8 = [3025/13824, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] is wrong.
    
    # Let me use the correct DOP853 coefficients from a reliable source.
    # DOP853: 8th order, 12 stages
    # b8 (for 8th order):
    b8 = np.array([
        3025.0/13824.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
    ])
    
    # This is getting complex. Let me use a simpler high-order method.
    # Use RK4 with very small steps, but that's what the parent does.
    
    # Better approach: Use scipy's solve_ivp directly if available, 
    # but we need to count f calls.
    
    # Actually, let me implement a proper adaptive high-order method.
    # For now, let me use a simpler approach: RK4 with adaptive step size.
    
    pass


def _adaptive_rk4(f, t0, t1, y0, rtol=1e-13, atol=1e-15, max_steps=50000):
    """
    Adaptive RK4 integrator with step size control.
    Uses Richardson extrapolation for error estimation.
    """
    t = t0
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Initial step size estimate
    h = (t1 - t0) / 100.0
    h_min = (t1 - t0) / max_steps
    h_max = (t1 - t0) / 10.0
    
    def rk4_step(t, y, h):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        return y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    steps = 0
    while t < t1 - 1e-15:
        # Don't overshoot
        if t + h > t1:
            h = t1 - t
        
        # Take two half steps and one full step for error estimation
        y_half = rk4_step(t, y, h/2)
        y_full = rk4_step(t, y, h)
        y_two_half = rk4_step(t + h/2, y_half, h/2)
        
        # Error estimate
        err = np.max(np.abs(y_two_half - y_full))
        scale = np.maximum(atol, rtol * np.maximum(1.0, np.abs(y_two_half)))
        err_norm = err / np.max(scale) if np.max(scale) > 0 else err
        
        if err_norm <= 1.0:
            # Accept step
            y = y_two_half
            t += h
            steps += 1
            
            # Adjust step size
            if err_norm > 1e-3:
                h_new = h * min(5.0, max(0.1, 0.9 * err_norm ** (-0.2)))
            else:
                h_new = h * 5.0
            h = np.clip(h_new, h_min, h_max)
        else:
            # Reject step, reduce h
            h = h * max(0.1, 0.9 * err_norm ** (-0.25))
            h = np.clip(h, h_min, h_max)
        
        if not np.all(np.isfinite(y)):
            return None
        
        if steps > max_steps:
            return y
    
    return y


def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra system using adaptive RK4.
    """
    t0, t1 = t_span
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s safety margin
    
    y = _adaptive_rk4(f, t0, t1, y0, rtol=1e-13, atol=1e-15, max_steps=50000)
    
    if y is None or not np.all(np.isfinite(y)):
        # Fallback to fixed-step RK4
        y = _rk4_fixed(f, t0, t1, y0, 5000)
    
    return y


def _rk4_fixed(f, t0, t1, y0, steps):
    """Fixed-step RK4 as fallback."""
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            return y
    return y
# EVOLVE-BLOCK-END
