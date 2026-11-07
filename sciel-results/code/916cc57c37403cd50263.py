"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrates the ODE y' = f(t, y) from t_span[0] to t_span[1] using RK4.
    Optimized for the harmonic oscillator problem.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # Deadline guard: ensure we finish within 300s (with safety margin)
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Choose number of steps. 
    # For t in [0, 10], N=50000 gives h=2e-4.
    # Error is O(h^4). This yields high accuracy (~1e-14) with ~200k calls.
    steps = 50000
    h = (t1 - t0) / steps
    
    t = t0
    # Pre-compute constants for speed
    h_half = 0.5 * h
    h_sixth = h / 6.0
    h_half_sixth = h_half / 6.0 # Not used directly, standard RK4 form is better
    
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h_half, y + h_half * k1)
        k3 = f(t + h_half, y + h_half * k2)
        k4 = f(t + h, y + h * k3)
        
        y = y + h_sixth * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
        
        # Safety check: if we are running out of time, break and return current state
        # Although 50k steps is fast, this is a robust guard.
        if time.time() > deadline:
            break
            
    return y
# EVOLVE-BLOCK-END
