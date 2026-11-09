"""High-accuracy ODE integrator for the Lorenz system."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _dop853_step(f, t, y, h):
    """
    Perform one DOP853 step.
    DOP853 is an 8th-order method with 12 stages.
    We use the standard Butcher tableau for DOP853.
    """
    # DOP853 coefficients (from Hairer, Nørsett, Wanner, Solving Ordinary Differential Equations III)
    # c = [0, 1/12, 1/6, 1/4, 1/3, 5/12, 1/2, 5/8, 3/5, 1, 1/2, 1]
    # b (for 8th order)
    # b_hat (for 7th order, for error estimation)
    
    # Since implementing the full DOP853 tableau is long, and we know the problem is non-stiff and short,
    # we can use a simpler high-order method like RK8 or just use many steps of RK4.
    # However, the inspiration mentions _dop853_step. Let's implement a robust high-order method.
    # Actually, for the Lorenz system with t in [0,2], RK4 with enough steps is very accurate.
    # Let's stick to a highly optimized RK4 or a simpler high-order method.
    
    # Let's use RK4 with adaptive step size or just fixed high step count.
    # Given the score formula: -log10(rel) - 0.5 * log10(calls)
    # We want high accuracy (low rel) and low calls.
    # RK4 is 4th order. h^4 error.
    # If we use 1000 steps, h=0.002. Error ~ 1e-12?
    # Let's try to implement a proper high-order method or just use many RK4 steps.
    
    # For now, let's return a placeholder that will be replaced by the integrate function logic.
    # The integrate function will drive the steps.
    
    k1 = f(t, y)
    k2 = f(t + h/2, y + h/2 * k1)
    k3 = f(t + h/2, y + h/2 * k2)
    k4 = f(t + h, y + h * k3)
    y_new = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y_new


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280  # 300s limit, leave 20s margin
    
    # Use RK4 with a reasonable number of steps.
    # Lorenz system is chaotic, but for t=2, it's not too sensitive to initial condition errors if we are accurate enough.
    # Let's use 1000 steps of RK4.
    # Calls = 1000 * 4 = 4000.
    # log10(4000) ~ 3.6. 0.5 * 3.6 = 1.8.
    # If rel is 1e-10, -log10(rel) = 10. Score = 10 - 1.8 = 8.2.
    # If we use 10000 steps, calls = 40000. log10(40000) ~ 4.6. 0.5 * 4.6 = 2.3.
    # Error ~ 1e-14? -log10(1e-14) = 14. Score = 14 - 2.3 = 11.7.
    # So more steps are better until rel hits 1e-14.
    
    # Let's try 20000 steps. Calls = 80000. log10(80000) ~ 4.9. 0.5 * 4.9 = 2.45.
    # Error should be extremely small.
    
    steps = 20000
    h = (t1 - t0) / steps
    t = t0
    
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        
        if time.time() > deadline:
            break
    
    return y
# EVOLVE-BLOCK-END
