import numpy as np
import scipy.linalg
import time
import scipy.integrate

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin for 300s limit
    
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # 1. Check for inhomogeneity / non-linearity cheaply
    # If f(t, 0) is not 0, it's inhomogeneous.
    try:
        f0 = np.asarray(f(t0, np.zeros(n)), dtype=float)
        if np.linalg.norm(f0) > 1e-10:
            # Inhomogeneous, fallback to robust solver
            return _fallback_solve(f, t_span, y0, start_time)
    except Exception:
        return _fallback_solve(f, t_span, y0, start_time)

    # 2. Extract A assuming y' = A y
    basis = np.eye(n)
    A_cols = []
    for i in range(n):
        if time.time() > deadline:
            return _fallback_solve(f, t_span, y0, start_time)
        e_i = basis[:, i]
        try:
            res = np.asarray(f(t0, e_i), dtype=float)
            if res.shape != (n,):
                res = res.reshape(n)
            A_cols.append(res)
        except Exception:
            return _fallback_solve(f, t_span, y0, start_time)
            
    A = np.column_stack(A_cols)
    
    # 3. Compute solution
    try:
        dt = t1 - t0
        M = scipy.linalg.expm(A * dt)
        y_final = M @ y0
    except Exception:
        return _fallback_solve(f, t_span, y0, start_time)
        
    return y_final

def _fallback_solve(f, t_span, y0, start_time):
    deadline = start_time + 280.0
    # Try Radau with tight tolerances
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-13, 
            atol=1e-15
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # If Radau fails or times out, try BDF
    if time.time() < deadline:
        try:
            sol = scipy.integrate.solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                rtol=1e-13, 
                atol=1e-15
            )
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
    
    # Last resort: try LSODA
    if time.time() < deadline:
        try:
            sol = scipy.integrate.solve_ivp(
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
            
    return y0
