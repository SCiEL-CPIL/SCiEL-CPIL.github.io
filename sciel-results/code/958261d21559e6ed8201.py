"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_reference(f, t_span, y0):
    """
    Compute a high-accuracy reference solution using Radau IIA (order 5)
    with Newton iteration for the implicit stage equations.
    """
    t0, t1 = t_span
    n = len(y0)
    y = np.array(y0, dtype=float)
    
    # Radau IIA order 5 coefficients
    # But Radau IIA order 5 is complex to implement correctly from scratch with Newton.
    # Instead, let's use a high-order explicit method with very small steps, 
    # or a simpler implicit method like Backward Euler with many Newton steps,
    # or Trapezoidal rule.
    
    # Actually, for Van der Pol with mu=5, the system is stiff.
    # Explicit methods require very small steps.
    # Implicit methods (like Backward Euler or Radau) are better.
    
    # Let's implement a simple implicit method: Backward Euler with Newton.
    # But Backward Euler is only 1st order. To get high accuracy, we need many steps.
    # Or we can use a higher-order implicit method.
    
    # Alternative: Use a high-order explicit method (RK4) with enough steps to be accurate.
    # For Van der Pol with mu=5, the solution has fast transients initially, then settles to a limit cycle.
    # The limit cycle is periodic with period ~ 2*pi/sqrt(1) ~ 6.28? No, for large mu, period is ~ 2.16*mu.
    # For mu=5, period ~ 10.8. So in t in [0, 10], we see almost one full period.
    # The initial transient is fast (time scale ~ 1/mu = 0.2).
    
    # Let's try RK4 with a large number of steps, say 10000 or 20000.
    # This will use 4 * 20000 = 80000 calls, which is well within the 2M limit.
    # And RK4 is 4th order, so with h=10/20000=5e-4, the error should be very small.
    
    # Let's implement RK4 with adaptive step size or just a fixed large number of steps.
    
    steps = 50000
    h = (t1 - t0) / steps
    t = t0
    y = np.array(y0, dtype=float)
    
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        
        if not np.all(np.isfinite(y)):
            break
    
    return y


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator using a high-accuracy method.
    We'll use RK4 with a large number of steps to ensure accuracy.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Use RK4 with 50000 steps for high accuracy
    # This gives 200000 function calls, well within the 2M limit
    h = (t1 - t0) / 50000
    t = t0
    y = y0.copy()
    
    for _ in range(50000):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        
        if not np.all(np.isfinite(y)):
            break
    
    return y
# EVOLVE-BLOCK-END
