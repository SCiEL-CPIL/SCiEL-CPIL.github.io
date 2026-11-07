"""Rewritten ODE integrator for the fixed Lorenz IVP using scipy DOP853."""
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    sol = solve_ivp(
        f,
        t_span,
        y0,
        method='DOP853',
        rtol=1e-11,
        atol=1e-13,
        dense_output=False
    )
    
    if sol.success:
        return sol.y[:, -1]
    
    # Fallback: high-step RK4 if solve_ivp fails (unlikely)
    # But solve_ivp with DOP853 is very robust for this problem.
    # If it fails, we try a simple high-resolution RK4.
    steps = 10000
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y
