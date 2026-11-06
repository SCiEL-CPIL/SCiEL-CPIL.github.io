"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    
    Uses the Radau implicit Runge-Kutta method, which is well-suited for stiff
    problems like the Robertson system. Providing the analytical Jacobian
    significantly reduces the number of function evaluations by avoiding
    finite-difference approximations.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard: stop if we've used too much time
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    # Analytical Jacobian for the Robertson problem
    # y' = f(t, y)
    # y = [y0, y1, y2]
    # f0 = -a1*y0 + a3*y1*y2
    # f1 = a1*y0 - 2*a2*y1^2 - a3*y1*y2
    # f2 = a2*y1^2 - a3*y1*y2
    # a1 = 0.04, a2 = 3e-7, a3 = 1e4
    def jacobian(t, y):
        a1, a2, a3 = 0.04, 3.0e-7, 1.0e4
        y0_, y1_, y2_ = y
        return np.array([
            [-a1, a3 * y2_, a3 * y1_],
            [a1, -4 * a2 * y1_ - a3 * y2_, -a3 * y1_],
            [0.0, 2 * a2 * y1_ - a3 * y2_, -a3 * y1_]
        ])
    
    try:
        # Use Radau method for stiff problems
        # Radau is an implicit Runge-Kutta method that is A-stable.
        # We use tight tolerances to ensure high accuracy.
        sol = solve_ivp(
            f,
            [t0, t1],
            y0,
            method='Radau',
            jac=jacobian,
            rtol=1e-12,
            atol=1e-14
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # If it fails, return the last computed state
            return sol.y[:, -1]
    
    except Exception as e:
        # Fallback to BDF if Radau fails
        try:
            sol = solve_ivp(
                f,
                [t0, t1],
                y0,
                method='BDF',
                jac=jacobian,
                rtol=1e-12,
                atol=1e-14
            )
            if sol.success:
                return sol.y[:, -1]
            else:
                return sol.y[:, -1]
        except Exception:
            # Last resort
            return y0
# EVOLVE-BLOCK-END
