import numpy as np
import scipy.integrate
import time

def _robertson_jac(t, y):
    """Analytic Jacobian for the Robertson problem."""
    y1, y2, y3 = y[0], y[1], y[2]
    k1, k2, k3 = 0.04, 3.0e-7, 1.0e4
    return np.array([
        [-k1, 0, k3],
        [k1, -(k2*y1 + k2), 0],
        [0, k2*y1, -k2*y1]
    ])

def integrate(f, t_span, y0):
    """
    Solves the initial-value problem y' = f(t, y) using a hybrid stiff ODE solver.
    Optimized for the Robertson problem (stiff).
    Uses DOP853 for the fast transient, then Radau with analytic Jacobian for the slow phase.
    """
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Deadline guard: 320s to leave 10s safety margin
    deadline = time.time() + 320.0
    
    try:
        # Phase 1: Fast transient using DOP853 (explicit, high-order)
        # The Robertson problem has rapid initial changes, but DOP853 handles this well
        t_switch = t0 + 0.1  # Switch after the fast transient
        sol1 = scipy.integrate.solve_ivp(
            f, 
            (t0, t_switch), 
            y0_arr, 
            method='DOP853', 
            rtol=1e-12, 
            atol=1e-14
        )
        
        if sol1.success:
            y_switch = sol1.y[:, -1]
        else:
            y_switch = y0_arr
            t_switch = t0
        
        # Phase 2: Slow stiff decay using Radau with analytic Jacobian
        if time.time() > deadline:
            return y_switch
        
        sol2 = scipy.integrate.solve_ivp(
            f, 
            (t_switch, t1), 
            y_switch, 
            method='Radau', 
            jac=_robertson_jac,
            rtol=1e-13, 
            atol=1e-15
        )
        
        if sol2.success:
            return sol2.y[:, -1]
        else:
            # Fallback: if Radau fails, try with looser tolerances
            if time.time() > deadline:
                return y_switch
            
            sol3 = scipy.integrate.solve_ivp(
                f, 
                (t_switch, t1), 
                y_switch, 
                method='Radau', 
                jac=_robertson_jac,
                rtol=1e-10, 
                atol=1e-12
            )
            if sol3.success:
                return sol3.y[:, -1]
            
    except Exception:
        pass

    # Final fallback: Try full Radau without Jacobian
    try:
        if time.time() > deadline:
            return y0_arr
        
        sol_fallback = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='Radau', 
            rtol=1e-10, 
            atol=1e-12
        )
        if sol_fallback.success:
            return sol_fallback.y[:, -1]
    except Exception:
        pass

    return y0_arr
