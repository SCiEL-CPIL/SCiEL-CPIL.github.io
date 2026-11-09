"""ODE solver: Convection-diffusion 100-point - adaptive high-order RK with stiff switching."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import solve
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    dt_total = t1 - t0

    # Try a few steps with RK4 to detect stiffness
    # If the solution changes very rapidly, we're in stiff regime
    h_probe = dt_total / 100
    t = t0
    y = y0.copy()
    
    # Do a few RK4 steps to see behavior
    def rk4_step(t, y, h):
        k1 = np.asarray(f(t, y), dtype=float)
        k2 = np.asarray(f(t + h/2, y + h/2 * k1), dtype=float)
        k3 = np.asarray(f(t + h/2, y + h/2 * k2), dtype=float)
        k4 = np.asarray(f(t + h, y + h * k3), dtype=float)
        return y + h/6.0 * (k1 + 2*k2 + 2*k3 + k4)
    
    # Check if the problem is stiff by comparing RK4 with smaller steps
    y_rk4 = y.copy()
    for _ in range(10):
        y_rk4 = rk4_step(t, y_rk4, h_probe)
        t += h_probe
    
    t2 = t0
    y_rk4_fine = y0.copy()
    for _ in range(20):
        y_rk4_fine = rk4_step(t2, y_rk4_fine, h_probe/2)
        t2 += h_probe/2
    
    diff = np.max(np.abs(y_rk4 - y_rk4_fine))
    scale = max(1.0, np.max(np.abs(y_rk4)))
    relative_change = diff / scale
    
    if relative_change > 0.1:
        # Stiff problem: use implicit Radau method with tight tolerances
        sol = solve_ivp(f, [t0, t1], y0, method='Radau', 
                        rtol=1e-13, atol=1e-15, max_step=dt_total/2000)
        if sol.success:
            return sol.y[:, -1]
        # Fallback to BDF
        sol = solve_ivp(f, [t0, t1], y0, method='BDF', 
                        rtol=1e-13, atol=1e-15, max_step=dt_total/2000)
        if sol.success:
            return sol.y[:, -1]
        # Last resort: many small RK4 steps
        steps = 50000
        h = dt_total / steps
        t = t0
        y = y0.copy()
        for _ in range(steps):
            y = rk4_step(t, y, h)
            t += h
        return y
    else:
        # Non-stiff: use high-order explicit RK with adaptive stepping
        # Use RK45 (Bogacki-Shampine) with very tight tolerances
        sol = solve_ivp(f, [t0, t1], y0, method='RK45', 
                        rtol=1e-14, atol=1e-16, max_step=dt_total/100)
        if sol.success:
            return sol.y[:, -1]
        # Fallback: many RK4 steps
        steps = 10000
        h = dt_total / steps
        t = t0
        y = y0.copy()
        for _ in range(steps):
            y = rk4_step(t, y, h)
            t += h
        return y
# EVOLVE-BLOCK-END
