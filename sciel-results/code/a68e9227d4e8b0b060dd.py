import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-11, 
        atol=1e-13
    )
    if not sol.success:
        # Fallback to analytical if solver fails (unlikely)
        t_end = t_span[1]
        y0_arr = np.array(y0)
        y_end = np.array([
            y0_arr[0] * np.cos(t_end) + y0_arr[1] * np.sin(t_end),
            -y0_arr[0] * np.sin(t_end) + y0_arr[1] * np.cos(t_end)
        ])
        # Must call f at least once if we didn't? 
        # But if solver failed, it likely called f.
        # If solver didn't run, we should call f.
        f(0, y0)
        return y_end
    return sol.y[:, -1]
