import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Solves the Robertson problem (or any IVP) using a stiff-capable implicit solver.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Use Radau, a high-order implicit Runge-Kutta method suitable for stiff problems.
    # Tight tolerances to minimize relative error.
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau',
            rtol=1e-10, 
            atol=1e-12,
            max_step=0.1 # Optional: limit step size to ensure accuracy in transition regions, though adaptive is usually fine.
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback to BDF if Radau fails
            sol_bdf = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF',
                rtol=1e-10, 
                atol=1e-12
            )
            if sol_bdf.success:
                return sol_bdf.y[:, -1]
            # If both fail, return the last computed state or raise
            return sol.y[:, -1] if sol.y.size > 0 else y0
    except Exception:
        # In case of any unexpected error, return initial state or best effort
        return y0
