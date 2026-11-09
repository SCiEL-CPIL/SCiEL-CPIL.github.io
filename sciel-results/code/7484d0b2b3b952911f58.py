"""ODE solver for Robertson problem using SciPy's Radau IIA with analytic Jacobian."""
import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrate the ODE y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns the final state y(t_end).
    """
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Analytic Jacobian for the Robertson problem
    def jac(t, y):
        y0, y1, y2 = y[0], y[1], y[2]
        J = np.zeros((3, 3))
        J[0, 0] = -0.04
        J[0, 1] = 1e4 * y2
        J[0, 2] = 1e4 * y1
        J[1, 0] = 0.04
        J[1, 1] = -1e4 * y2 - 6e7 * y1
        J[1, 2] = -1e4 * y1
        J[2, 0] = 0.0
        J[2, 1] = 6e7 * y1
        J[2, 2] = 0.0
        return J

    # Use Radau IIA (order 5) which is excellent for stiff problems
    # Tight tolerances to minimize error
    sol = solve_ivp(
        f, 
        t_span, 
        y0_arr, 
        method='Radau', 
        rtol=1e-10, 
        atol=1e-12, 
        jac=jac,
        max_step=0.1 # Optional: prevent huge steps if instability arises, though Radau is stable
    )
    
    if not sol.success:
        # Fallback to BDF if Radau fails for some reason
        sol = solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='BDF', 
            rtol=1e-10, 
            atol=1e-12, 
            jac=jac
        )
        
    return sol.y[:, -1]
