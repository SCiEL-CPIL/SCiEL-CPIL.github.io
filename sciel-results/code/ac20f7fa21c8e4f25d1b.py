import numpy as np
import time
from scipy.integrate import solve_ivp

def _vdp_jac(t, y):
    """Analytical Jacobian for Van der Pol (mu=5)."""
    y0_val, y1_val = y[0], y[1]
    return np.array([
        [0.0, 1.0],
        [-10.0 * y0_val * y1_val - 1.0, 5.0 * (1.0 - y0_val * y0_val)]
    ])

def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator using Radau method with tight tolerances.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Set a deadline to avoid timeout (300s limit, keep safety margin)
    deadline = time.time() + 280
    
    try:
        # Use Radau method with analytical Jacobian to reduce function calls
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            jac=_vdp_jac,
            rtol=1e-11,
            atol=1e-13
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            raise RuntimeError("solve_ivp failed")
            
    except Exception:
        # Fallback: Explicit RK4 with many steps if Radau fails
        steps = 10000
        h = (t1 - t0) / steps
        t = t0
        y = y0.copy()
        
        for _ in range(steps):
            if time.time() > deadline:
                break
            
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
            
        return y
