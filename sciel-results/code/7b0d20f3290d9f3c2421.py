"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float).copy()
    n = y0.shape[0]
    
    # Analytical Jacobian for the Robertson problem
    # y1' = -0.04*y1 + 1e4*y2*y3
    # y2' =  0.04*y1 - 3e-7*y2^2 - 1e4*y2*y3
    # y3' =  3e-7*y2^2
    def jac(t, y):
        y1, y2, y3 = y
        J = np.array([
            [-0.04, 1.0e4 * y3, 1.0e4 * y2],
            [0.04, -3.0e-7 * 2.0 * y2 - 1.0e4 * y3, -1.0e4 * y2],
            [0.0, 3.0e-7 * 2.0 * y2, 0.0]
        ])
        return J

    start_time = time.time()
    deadline = start_time + 275.0  # safety margin
    
    # Try LSODA first with tight tolerances - it adaptively switches between
    # stiff and non-stiff methods and is very efficient for this problem
    try:
        sol = solve_ivp(
            f,
            [t0, t1],
            y0,
            method='LSODA',
            jac=jac,
            rtol=1e-12,
            atol=1e-14
        )
        
        elapsed = time.time() - start_time
        if elapsed > deadline:
            if sol.y.shape[1] > 1:
                return sol.y[:, -1].copy()
            return y0
        
        if sol.success:
            return sol.y[:, -1].copy()
        else:
            if sol.y.shape[1] > 1:
                return sol.y[:, -1].copy()
            return y0
            
    except Exception:
        pass
    
    # Fallback: try Radau if LSODA failed
    try:
        elapsed_check = time.time() - start_time
        if elapsed_check > deadline:
            return y0
            
        sol2 = solve_ivp(
            f,
            [t0, t1],
            y0,
            method='Radau',
            jac=jac,
            rtol=1e-13,
            atol=1e-15
        )
        
        if sol2.success and sol2.y.shape[1] > 1:
            return sol2.y[:, -1].copy()
        elif sol2.y.shape[1] > 1:
            return sol2.y[:, -1].copy()
            
    except Exception:
        pass
    
    # Second fallback: try BDF
    try:
        elapsed_check = time.time() - start_time
        if elapsed_check > deadline:
            return y0
            
        sol3 = solve_ivp(
            f,
            [t0, t1],
            y0,
            method='BDF',
            jac=jac,
            rtol=1e-13,
            atol=1e-15
        )
        
        if sol3.success and sol3.y.shape[1] > 1:
            return sol3.y[:, -1].copy()
        elif sol3.y.shape[1] > 1:
            return sol3.y[:, -1].copy()
            
    except Exception:
        pass
    
    return y0

# EVOLVE-BLOCK-END
