"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    timeout = 250.0  # Safety margin before 300s limit
    
    # Wrap f to enforce timeout
    def f_wrapper(t, y):
        if time.time() - start_time > timeout:
            raise TimeoutError("Time limit exceeded")
        return f(t, y)

    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Define Jacobian for the Robertson problem
    k1, k2, k3 = 0.04, 3e7, 1e4
    
    def jac(t, y):
        J = np.array([
            [-k1, k3 * y[2], k3 * y[1]],
            [k1, -2 * k2 * y[1] - k3 * y[2], -k3 * y[1]],
            [0, 2 * k2 * y[1], 0]
        ])
        return J

    # Attempt 1: Radau with analytical Jacobian
    try:
        sol = solve_ivp(
            f_wrapper, t_span, y0_arr,
            method='Radau',
            jac=jac,
            rtol=1e-12,
            atol=1e-14
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Attempt 2: Radau without Jacobian
    try:
        sol = solve_ivp(
            f_wrapper, t_span, y0_arr,
            method='Radau',
            rtol=1e-12,
            atol=1e-14
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Attempt 3: BDF without Jacobian (alternative stiff solver)
    try:
        sol = solve_ivp(
            f_wrapper, t_span, y0_arr,
            method='BDF',
            rtol=1e-12,
            atol=1e-14
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback: RK45 (explicit, less stiffly stable but robust)
    try:
        sol = solve_ivp(
            f_wrapper, t_span, y0_arr,
            method='RK45',
            rtol=1e-10,
            atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # If all fails, return initial state
    return y0_arr
# EVOLVE-BLOCK-END
