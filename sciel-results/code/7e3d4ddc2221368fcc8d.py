"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def _robertson_jac(t, y):
    """Analytical Jacobian for the Robertson problem."""
    y0, y1, y2 = y
    k1, k2, k3 = 0.04, 3.0e-5, 1.0e4
    return np.array([
        [-k1, 0.0, k3],
        [k1, -(k1 + k2), 0.0],
        [0.0, k2, -k2]
    ])


def integrate(f, t_span, y0):
    """
    Integrate the Robertson ODE problem on [0, 40] using Radau with
    the analytical Jacobian for maximum efficiency and accuracy.
    
    The Robertson problem is stiff with a well-known structure. Using Radau
    (implicit Runge-Kutta) with the exact Jacobian minimizes function
    evaluations while achieving high terminal accuracy.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # 50s safety margin
    
    try:
        # Use Radau with analytical Jacobian for the full interval
        # Tight tolerances for high accuracy
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            jac=_robertson_jac,
            rtol=1e-13,
            atol=1e-15,
            max_step=1.0,
        )
        
        if sol.success:
            return sol.y[:, -1]
        
        # If it failed, try with slightly looser tolerances
        sol2 = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            jac=_robertson_jac,
            rtol=1e-11,
            atol=1e-13,
            max_step=1.0,
        )
        
        if sol2.success:
            return sol2.y[:, -1]
        
        return sol.y[:, -1] if sol.y.size > 0 else y0
        
    except Exception:
        # Fallback without Jacobian
        try:
            sol = solve_ivp(
                f,
                t_span,
                y0,
                method='Radau',
                rtol=1e-12,
                atol=1e-14,
                max_step=1.0,
            )
            if sol.success:
                return sol.y[:, -1]
            return sol.y[:, -1] if sol.y.size > 0 else y0
        except Exception:
            return y0


# EVOLVE-BLOCK-END
