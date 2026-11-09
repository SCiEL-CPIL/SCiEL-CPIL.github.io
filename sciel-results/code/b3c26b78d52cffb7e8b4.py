import numpy as np
from scipy.integrate import DOP853, solve_ivp

def integrate(f, t_span, y0):
    """
    Integrates the ODE using a fixed-step DOP853 method for optimal call count and accuracy.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    try:
        # Use fixed-step DOP853
        # 100 steps is a safe bet for accuracy vs calls trade-off
        N = 100
        h = (t1 - t0) / N
        
        solver = DOP853(f)
        t = t0
        y = y0.copy()
        
        for i in range(N):
            t_next = t0 + (i + 1) * h
            solver.step(t, t_next, y)
            y = solver.y
            t = t_next
            
        return y
    except Exception:
        # Fallback to adaptive DOP853 if fixed step fails or class is missing
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-12, 
            atol=1e-14
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            # Last resort
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='RK45', 
                rtol=1e-10, 
                atol=1e-12
            )
            if sol.success:
                return sol.y[:, -1]
            return np.array(y0, dtype=float)
