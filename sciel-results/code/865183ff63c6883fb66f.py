import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Solves the ODE y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Uses a high-order adaptive Runge-Kutta method (DOP853) to balance accuracy and function evaluations.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # If the time span is zero, return the initial condition
    if t0 == t1:
        return y0

    # Use DOP853 (8th order Dormand-Prince) for high accuracy with relatively few steps
    # rtol and atol are set very tight to minimize relative error
    # max_step is left to the adaptive controller, but we can constrain it if needed
    # For a 10x10 heat equation on [0, 0.05], the dynamics are smooth.
    
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853',
            rtol=1e-13, 
            atol=1e-15,
            max_step=(t1 - t0) / 50  # Allow up to 50 steps initially, adaptive will refine
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback if DOP853 fails (unlikely for smooth heat equation)
            # Try Radau (implicit, good for stiff)
            sol_radau = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='Radau',
                rtol=1e-13, 
                atol=1e-15
            )
            if sol_radau.success:
                return sol_radau.y[:, -1]
            else:
                # Last resort: simple RK4 with many steps
                h = (t1 - t0) / 1000
                t, y = t0, y0.copy()
                for _ in range(1000):
                    k1 = f(t, y)
                    k2 = f(t + h/2, y + h/2 * k1)
                    k3 = f(t + h/2, y + h/2 * k2)
                    k4 = f(t + h, y + h * k3)
                    y = y + h/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)
                    t += h
                return y
                
    except Exception:
        # Safety fallback
        h = (t1 - t0) / 1000
        t, y = t0, y0.copy()
        for _ in range(1000):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)
            t += h
        return y
