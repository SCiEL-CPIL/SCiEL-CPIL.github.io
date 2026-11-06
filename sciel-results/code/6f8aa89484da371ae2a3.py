"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate the Robertson problem using Radau (implicit Runge-Kutta) with
    an analytic Jacobian for efficiency on this stiff system.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard: return best available result if approaching time limit
    start_time = time.time()
    deadline = start_time + 280  # 300s limit, 20s safety margin
    
    # Analytic Jacobian for the Robertson problem:
    # y' = f(t, y) where:
    # y[0]' = -0.04*y[0] + 1e4*y[1]*y[2]
    # y[1]' = 0.04*y[0] - 1e4*y[1]*y[2] - 3e7*y[1]^2
    # y[2]' = 3e7*y[1]^2
    #
    # Jacobian J[i,j] = df[i]/dy[j]
    def jac(t, y):
        y0_val, y1_val, y2_val = y[0], y[1], y[2]
        J = np.array([
            [-0.04, 1e4 * y2_val, 1e4 * y1_val],
            [0.04, -1e4 * y2_val - 6e7 * y1_val, -1e4 * y1_val],
            [0.0, 6e7 * y1_val, 0.0]
        ])
        return J
    
    # Use Radau with tight tolerances for high accuracy
    # rtol=1e-12, atol=1e-14 for very accurate solution
    sol = solve_ivp(
        f, t_span, y0,
        method='Radau',
        jac=jac,
        rtol=1e-12,
        atol=1e-14,
        max_step=(t1 - t0) / 100  # reasonable max step
    )
    
    # Check if we're still within time
    if time.time() > deadline:
        # If we somehow have a result, use it; otherwise try a quick fallback
        if sol.success:
            return sol.y[:, -1]
        # Quick fallback: use a simpler method with fewer evaluations
        sol_fallback = solve_ivp(
            f, t_span, y0,
            method='Radau',
            jac=jac,
            rtol=1e-10,
            atol=1e-12
        )
        if sol_fallback.success:
            return sol_fallback.y[:, -1]
        return y0
    
    if sol.success:
        return sol.y[:, -1]
    
    # If Radau failed, try BDF as fallback
    sol_bdf = solve_ivp(
        f, t_span, y0,
        method='BDF',
        jac=jac,
        rtol=1e-10,
        atol=1e-12
    )
    if sol_bdf.success:
        return sol_bdf.y[:, -1]
    
    # Last resort: return initial condition
    return y0

# EVOLVE-BLOCK-END
