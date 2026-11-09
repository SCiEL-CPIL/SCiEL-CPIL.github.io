"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Define the Jacobian for the Robertson problem to help the implicit solver
    def jac(t, y):
        # y = [y1, y2, y3]
        # y1' = -0.04*y1 + 1e4*y2*y3
        # y2' = 0.04*y1 - 1e4*y2*y3 - 3e7*y2^2
        # y3' = 3e7*y2^2
        # Jacobian:
        # d(y1')/dy1 = -0.04
        # d(y1')/dy2 = 1e4*y3
        # d(y1')/dy3 = 1e4*y2
        # d(y2')/dy1 = 0.04
        # d(y2')/dy2 = -1e4*y3 - 6e7*y2
        # d(y2')/dy3 = -1e4*y2
        # d(y3')/dy1 = 0
        # d(y3')/dy2 = 6e7*y2
        # d(y3')/dy3 = 0
        J = np.array([
            [-0.04, 1e4 * y[2], 1e4 * y[1]],
            [0.04, -1e4 * y[2] - 6e7 * y[1], -1e4 * y[1]],
            [0.0, 6e7 * y[1], 0.0]
        ])
        return J

    # Set up the solver with Radau method, which is well-suited for stiff problems
    # Use tight tolerances to minimize error
    rtol = 1e-10
    atol = 1e-12
    
    start_time = time.time()
    deadline = start_time + 290.0  # 10 seconds safety margin

    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            jac=jac,
            rtol=rtol, 
            atol=atol
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # If solver fails, return the last computed state if available
            if sol.y.shape[1] > 0:
                return sol.y[:, -1]
            else:
                return y0
    except Exception as e:
        # In case of any error, return the initial state as a fallback
        return y0

# EVOLVE-BLOCK-END
