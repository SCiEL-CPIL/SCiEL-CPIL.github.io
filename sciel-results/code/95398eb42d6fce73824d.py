# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 280.0
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    from scipy.integrate import solve_ivp
    
    # Try high accuracy first
    # DOP853 is 8th order, very efficient
    sol = solve_ivp(f, (t0, t1), y0, method='DOP853', rtol=1e-14, atol=1e-16, dense_output=False)
    
    if time.time() > deadline - 5.0:
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
        return _fallback(f, t0, t1, y0)
    
    if not sol.success or not np.all(np.isfinite(sol.y[:, -1])):
        # Fallback to slightly looser tolerance
        sol2 = solve_ivp(f, (t0, t1), y0, method='DOP853', rtol=1e-12, atol=1e-14, dense_output=False)
        if sol2.success and np.all(np.isfinite(sol2.y[:, -1])):
            return sol2.y[:, -1]
        return _fallback(f, t0, t1, y0)
    
    return sol.y[:, -1]

def _fallback(f, t0, t1, y0):
    """High-accuracy RK4 fallback"""
    n_steps = 100000
    h = (t1 - t0) / n_steps
    t = t0
    y = np.array(y0, dtype=float)
    for _ in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y
# EVOLVE-BLOCK-END
