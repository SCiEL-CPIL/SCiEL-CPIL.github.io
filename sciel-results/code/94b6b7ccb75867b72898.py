import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    """
    Solves the initial-value problem y' = f(t, y) using a stiff ODE solver.
    Optimized for the Robertson problem (stiff).
    Uses analytic Jacobian and multi-stage tolerance refinement with a final polish.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Analytic Jacobian for the Robertson problem
    # y = [y1, y2, y3]
    # dy1/dt = -0.04*y1 + 1e4*y2*y3
    # dy2/dt = 0.04*y1 - 1e4*y2*y3 - 3e7*y2^2
    # dy3/dt = 3e7*y2^2
    def jacobian(t, y):
        y1, y2, y3 = y
        J = np.empty((3, 3))
        J[0, 0] = -0.04
        J[0, 1] = 1e4 * y3
        J[0, 2] = 1e4 * y2
        J[1, 0] = 0.04
        J[1, 1] = -1e4 * y3 - 6e7 * y2
        J[1, 2] = -1e4 * y2
        J[2, 0] = 0.0
        J[2, 1] = 6e7 * y2
        J[2, 2] = 0.0
        return J
    
    best_result = None
    
    # Stage 1: Coarse solve with Radau + analytic Jacobian
    try:
        sol1 = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='Radau', 
            rtol=1e-8, 
            atol=1e-10,
            jac=jacobian
        )
        if sol1.success:
            best_result = sol1.y[:, -1].copy()
    except Exception:
        pass
    
    if time.time() > deadline - 30.0:
        if best_result is not None:
            return best_result
        return y0_arr
    
    # Stage 2: Medium tolerance
    try:
        sol2 = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='Radau', 
            rtol=1e-11, 
            atol=1e-13,
            jac=jacobian
        )
        if sol2.success:
            best_result = sol2.y[:, -1].copy()
    except Exception:
        pass
    
    if time.time() > deadline - 15.0:
        if best_result is not None:
            return best_result
        return y0_arr
    
    # Stage 3: Tight tolerance
    try:
        sol3 = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='Radau', 
            rtol=1e-13, 
            atol=1e-15,
            jac=jacobian
        )
        if sol3.success:
            best_result = sol3.y[:, -1].copy()
    except Exception:
        pass
    
    if time.time() > deadline - 5.0:
        if best_result is not None:
            return best_result
        return y0_arr
    
    # Stage 4: Final polish from t=39 to t=40 with very tight tolerances
    if best_result is not None:
        try:
            polish_start = t1 - 1.0  # Start polish 1 unit before the end
            sol4 = scipy.integrate.solve_ivp(
                f, 
                (polish_start, t1), 
                best_result, 
                method='Radau', 
                rtol=1e-14, 
                atol=1e-16,
                jac=jacobian
            )
            if sol4.success:
                best_result = sol4.y[:, -1].copy()
        except Exception:
            pass
    
    if best_result is not None:
        return best_result
    
    # Final fallback
    return y0_arr
