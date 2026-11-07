import numpy as np
import scipy.integrate

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Tighten tolerances to minimize relative error (rel)
    # Reference uses rtol=1e-11, atol=1e-13.
    # We use rtol=1e-13, atol=1e-15 to push rel closer to the floor.
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='DOP853', 
            rtol=1e-13, 
            atol=1e-15
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
        
    # Fallback: High precision RK4
    # Increase steps to ensure accuracy if solver fails
    n_steps = 50000
    h = (t1 - t0) / n_steps
    t = t0
    y = y0_arr.copy()
    for _ in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y
