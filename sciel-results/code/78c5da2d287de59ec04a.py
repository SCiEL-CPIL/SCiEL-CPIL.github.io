"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.integrate

def integrate(f, t_span, y0):
    """Main integration function using scipy for high accuracy with low call count."""
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    deadline = time.time() + 300.0
    
    # Use scipy's DOP853 (8th order) for high accuracy with fewer steps
    # This method is very efficient for smooth problems
    result = scipy.integrate.solve_ivp(
        f, t_span, y0,
        method='DOP853',
        rtol=1e-12,
        atol=1e-14,
        max_step=(t1 - t0) / 100.0  # Reasonable max step to ensure accuracy
    )
    
    if result.success and result.t[-1] == t1:
        return result.y[:, -1]
    
    # Check deadline before fallback
    if time.time() > deadline - 5.0:
        return result.y[:, -1] if result.y.size > 0 else y0
    
    # Fallback: use Radau for stiff problems
    result = scipy.integrate.solve_ivp(
        f, t_span, y0,
        method='Radau',
        rtol=1e-12,
        atol=1e-14
    )
    
    if result.success and result.t[-1] == t1:
        return result.y[:, -1]
    
    # Check deadline before RK4
    if time.time() > deadline - 5.0:
        return result.y[:, -1] if result.y.size > 0 else y0
    
    # Last resort: fixed-step RK4
    steps = 10000
    h = (t1 - t0) / steps
    t = t0
    y = y0.copy()
    for i in range(steps):
        if time.time() > deadline - 2.0:
            break
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
    
    return y
# EVOLVE-BLOCK-END
