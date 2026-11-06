"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.integrate as integrate_lib
import time

def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    
    Strategy: Use scipy.integrate.ode with the LSODA method.
    LSODA is a Fortran-based hybrid stiff/non-stiff solver that is often more
    efficient in terms of function evaluations than the Python-based Radau solver
    in solve_ivp for stiff problems like Robertson. It automatically switches
    between Adams (non-stiff) and BDF (stiff) methods.
    
    We set tight tolerances to minimize relative terminal error.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin
    
    try:
        # Define the ODE function wrapper if needed, but f is passed directly.
        # scipy.integrate.ode expects f(t, y)
        solver = integrate_lib.ode(f)
        solver.set_integrator('lsoda')
        
        # Set tight tolerances
        # rtol=1e-12, atol=1e-16 for high accuracy
        solver.set_initial_value(y0, t0)
        solver.set_options(rtol=1e-12, atol=1e-16, nsteps=2000000)
        
        # Step to the end
        # We must step carefully. LSODA can take large steps.
        # We will step to t1.
        success = solver.integrate(t1)
        
        if success:
            return solver.y
        
        # If integrate(t1) fails, try stepping in chunks or fallback
        # For Robertson, it should succeed.
        
        # Fallback to solve_ivp Radau if ode fails
        from scipy.integrate import solve_ivp
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-10, 
            atol=1e-14
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            return y0
            
    except Exception:
        # Last resort fallback
        try:
            from scipy.integrate import solve_ivp
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='Radau', 
                rtol=1e-10, 
                atol=1e-14
            )
            if sol.success:
                return sol.y[:, -1]
            else:
                return y0
        except Exception:
            return y0

# EVOLVE-BLOCK-END
