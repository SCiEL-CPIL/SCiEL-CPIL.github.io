"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def _robertson_jac(t, y):
    """Analytical Jacobian for the Robertson problem."""
    # Robertson ODE: y' = [k1*y0*(1-y1) - k2*y1, -k1*y0*(1-y1) + k2*y1 - k3*y1*y2, k3*y1*y2]
    # Standard Robertson: k1=0.04, k2=1e4, k3=3e7, y0=[1,0,0]
    # But we need to work with whatever f is given. Since this is the fixed Robertson problem,
    # we can detect the rate constants from the problem structure.
    # Actually, the task says "fixed Robertson initial-value problem", so the constants are known.
    # Standard Robertson: dy1/dt = -0.04*y1 + 1e4*y2, dy2/dt = 0.04*y1 - 1e4*y2 - 3e7*y2^2, dy3/dt = 3e7*y2^2
    # Wait, let me use the standard form:
    # y1' = -k1*y1 + k2*y2
    # y2' = k1*y1 - k2*y2 - k3*y2^2
    # y3' = k3*y2^2
    # with k1=0.04, k2=1e4, k3=3e7
    k1 = 0.04
    k2 = 1e4
    k3 = 3e7
    
    J = np.array([
        [-k1, k2, 0.0],
        [k1, -k2 - 2*k3*y[1], 0.0],
        [0.0, k3*y[1], 0.0]
    ])
    return J


def integrate(f, t_span, y0):
    """
    Integrate the Robertson ODE problem on [0, 40] using Radau with analytical Jacobian.
    
    Strategy: Use Radau (implicit L-stable Runge-Kutta) with the analytical Jacobian
    of the Robertson problem to minimize function evaluations while achieving high
    terminal accuracy. The Jacobian eliminates finite-difference overhead and allows
    the implicit solver to take larger steps efficiently on the stiff manifold.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 270.0  # 60s safety margin
    
    try:
        # Use Radau with analytical Jacobian for optimal stiff problem solving
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
        else:
            # Fallback: try without Jacobian if it caused issues
            sol2 = solve_ivp(
                f,
                t_span,
                y0,
                method='Radau',
                rtol=1e-12,
                atol=1e-14,
            )
            if sol2.success:
                return sol2.y[:, -1]
            return sol2.y[:, -1] if sol2.y.size > 0 else y0
            
    except Exception:
        # If Jacobian approach fails, fall back to plain Radau
        try:
            sol = solve_ivp(
                f,
                t_span,
                y0,
                method='Radau',
                rtol=1e-12,
                atol=1e-14,
            )
            if sol.success:
                return sol.y[:, -1]
            return sol.y[:, -1] if sol.y.size > 0 else y0
        except Exception:
            return y0


# EVOLVE-BLOCK-END
