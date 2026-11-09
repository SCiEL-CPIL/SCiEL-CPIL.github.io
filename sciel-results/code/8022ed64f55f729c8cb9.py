"""ODE integrator for the Lorenz system using high-order Runge-Kutta."""
import numpy as np


def integrate(f, t_span, y0):
    """Return the state at the end of t_span for the Lorenz system.
    
    Uses a high-order fixed-step Runge-Kutta method with a moderate number
    of steps to achieve high accuracy with a controlled number of function calls.
    """
    t0, t1 = t_span
    y = np.asarray(y0, dtype=float).copy()
    
    # Use 8th-order Runge-Kutta (Dormand-Prince 8(5,3) style) with enough steps
    # For the Lorenz system over t in [0,2], we need sufficient resolution
    # Each step uses 12 function evaluations for a high-order method
    # With 200 steps, that's 2400 calls - well within the 2M limit
    # This should give high accuracy
    
    # Actually, let's use a simpler but effective approach:
    # 4th-order RK with many small steps, or a higher-order method with fewer steps
    
    # Let's use an 8th-order method with ~100 steps = 1200 calls
    # Or use adaptive stepping
    
    # For simplicity and reliability, use RK45 (Dormand-Prince) with tight tolerances
    # But implement it manually to control call count
    
    # Use a high-order fixed-step method
    n_steps = 500
    h = (t1 - t0) / n_steps
    t = t0
    
    # Use 4th-order Runge-Kutta with 500 steps = 2000 function calls
    # This should give good accuracy for this smooth problem
    for _ in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    
    return y
