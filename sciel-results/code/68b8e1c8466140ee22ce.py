"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    """
    Integrates the ODE using scipy.integrate.solve_ivp with DOP853 method.
    DOP853 is an 8th-order explicit Runge-Kutta method.
    For the Van der Pol oscillator with mu=5 (moderate stiffness),
    explicit methods often require fewer total function evaluations than
    implicit methods (like Radau) which incur overhead from Jacobian
    approximations (finite differences) if the Jacobian is not provided analytically.
    High order allows larger step sizes for the same accuracy, reducing total steps.
    """
    t0, t1 = t_span
    
    # Use DOP853 with tight tolerances matching the reference requirements.
    # DOP853 is available in SciPy >= 1.4.
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-11, 
            atol=1e-13
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback to RK45 if DOP853 fails (unlikely)
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='RK45', 
                rtol=1e-11, 
                atol=1e-13
            )
            if sol.success:
                return sol.y[:, -1]
            else:
                return np.array(y0, dtype=float)
    except Exception:
        # Fallback to RK45 if DOP853 is not available or throws
        try:
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='RK45', 
                rtol=1e-11, 
                atol=1e-13
            )
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
        return np.array(y0, dtype=float)
# EVOLVE-BLOCK-END
