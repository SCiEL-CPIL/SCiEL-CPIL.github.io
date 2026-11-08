"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin from 300s limit
    
    # Try SciPy's Radau method first (good for stiff problems)
    # We need to use scipy.integrate.solve_ivp
    try:
        from scipy.integrate import solve_ivp
        from scipy.integrate import Radau, BDF, LSODA
    except ImportError:
        # Fallback to manual RK45 if scipy not available
        def _rk45(f, t0, t1, y0, rtol=1e-10, atol=1e-12):
            y = np.array(y0, dtype=float)
            t = t0
            h = (t1 - t0) / 100.0  # initial step
            while t < t1:
                h = min(h, t1 - t)
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2*k1)
                k3 = f(t + h/2, y + h/2*k2)
                k4 = f(t + h, y + h*k3)
                k5 = f(t + h/5, y + h/5*k4)
                k6 = f(t + 3*h/5, y + 3*h/5*(k1 + 4/3*(k4-k1)))
                
                y_new = y + h*(k1 + 3*k2 + 3*k3 + k4)/8
                err = h*(k4 - k3)/32
                err_norm = np.max(np.abs(err) / (1 + np.abs(y_new)))
                
                if err_norm <= rtol:
                    y = y_new
                    t += h
                    h *= min(5.0, 0.9*(rtol/err_norm)**0.2) if err_norm > 0 else 5.0
                else:
                    h *= max(0.1, 0.9*(rtol/err_norm)**0.2)
                if time.time() > deadline:
                    break
            return y
        
        return _rk45(f, t0, t1, y0)
    
    # Use Radau with optimized tolerances and no max_step restriction
    # Radau is L-stable and good for stiff problems.
    # Removing max_step allows larger steps in stiff regions, reducing evaluations.
    try:
        sol = solve_ivp(
            f, t_span, y0,
            method='Radau',
            rtol=1e-10, atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Try BDF as fallback
    try:
        sol = solve_ivp(
            f, t_span, y0,
            method='BDF',
            rtol=1e-10, atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Try LSODA as fallback
    try:
        sol = solve_ivp(
            f, t_span, y0,
            method='LSODA',
            rtol=1e-10, atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Final fallback: RK45 with moderate tolerances
    def _rk45_fallback(f, t0, t1, y0, rtol=1e-10, atol=1e-12):
        y = np.array(y0, dtype=float)
        t = t0
        h = (t1 - t0) / 100.0
        while t < t1 and time.time() < deadline:
            h = min(h, t1 - t)
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2*k1)
            k3 = f(t + h/2, y + h/2*k2)
            k4 = f(t + h, y + h*k3)
            k5 = f(t + h/5, y + h/5*k4)
            k6 = f(t + 3*h/5, y + 3*h/5*(k1 + 4/3*(k4-k1)))
            
            y_new = y + h*(k1 + 3*k2 + 3*k3 + k4)/8
            err = h*(k4 - k3)/32
            err_norm = np.max(np.abs(err) / (1 + np.abs(y_new)))
            
            if err_norm <= rtol:
                y = y_new
                t += h
                h *= min(5.0, 0.9*(rtol/err_norm)**0.2) if err_norm > 0 else 5.0
            else:
                h *= max(0.1, 0.9*(rtol/err_norm)**0.2)
        return y
    
    return _rk45_fallback(f, t0, t1, y0)
# EVOLVE-BLOCK-END
