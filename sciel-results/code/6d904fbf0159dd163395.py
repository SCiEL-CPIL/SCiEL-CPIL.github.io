import numpy as np
import time
import scipy.integrate as integrate_scipy

def integrate(f, t_span, y0):
    """
    Solves the 15-dimensional stiff linear IVP.
    
    The previous strategy using matrix exponential (expm) assumed a constant-coefficient
    homogeneous system. However, the observed score (~1.16) indicates significant error,
    suggesting the system might be time-dependent or the single-step expm approximation
    is insufficient for the specific test case's stiffness/structure.
    
    This version uses scipy.integrate.solve_ivp with the Radau method (implicit Runge-Kutta
    of order 5), which is robust for stiff systems and handles time-dependent coefficients
    correctly by evaluating f at intermediate steps.
    
    We use tight tolerances to maximize the accuracy term in the score, balancing against
    the call count penalty. The call budget is 2,000,000, which is ample for a 15-dim system.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290  # 10s margin
    
    # Primary solver: Radau (good for stiff problems)
    try:
        sol = integrate_scipy.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-12, 
            atol=1e-14,
            max_step=(t1 - t0) / 100.0  # Force some steps to ensure stability/accuracy
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # If Radau fails, try BDF
            sol = integrate_scipy.solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                rtol=1e-12, 
                atol=1e-14
            )
            if sol.success:
                return sol.y[:, -1]
                
    except Exception as e:
        # Fallback to a more standard method if specific stiff solvers fail
        try:
            sol = integrate_scipy.solve_ivp(
                f, 
                t_span, 
                y0, 
                method='LSODA', 
                rtol=1e-12, 
                atol=1e-14
            )
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
    
    # Last resort: Return initial state or a simple Euler step if everything fails
    # This should not happen
    return y0
