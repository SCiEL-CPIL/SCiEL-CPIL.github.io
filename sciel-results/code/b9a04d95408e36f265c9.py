"""
ODE solver for the 10x10 Heat Equation using an implicit stiff solver.
"""
import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrates y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Uses an implicit Runge-Kutta method (Radau) suitable for stiff problems like the Heat Equation.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Radau is a 5th order implicit Runge-Kutta method, well-suited for stiff problems.
    # Tight tolerances to minimize relative terminal error.
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='Radau', 
        rtol=1e-10, 
        atol=1e-12
    )
    
    if not sol.success:
        # Fallback to BDF if Radau fails for some reason
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF', 
            rtol=1e-10, 
            atol=1e-12
        )
        
    # Return the final state
    return sol.y[:, -1]
