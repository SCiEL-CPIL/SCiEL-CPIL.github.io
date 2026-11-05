import numpy as np
import scipy.linalg
import scipy.integrate
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard: 300s limit, use 250s to be safe
    start_time = time.time()
    deadline = start_time + 250.0
    
    # Helper to check if we are out of time
    def timed_out():
        return time.time() > deadline

    # 1. Check for Linearity and Time-Invariance
    # We assume the heat equation is linear and autonomous.
    # If these checks fail, we fall back to a general-purpose ODE solver.
    
    # Use deterministic test vectors to avoid random overhead and potential edge cases
    y_test1 = np.ones(n)
    y_test2 = np.arange(1, n + 1, dtype=float)
    
    try:
        f_t0_y1 = f(t0, y_test1)
        f_t0_y2 = f(t0, y_test2)
        f_t0_y12 = f(t0, y_test1 + y_test2)
        
        # Check additivity: f(y1+y2) == f(y1) + f(y2)
        if not np.allclose(f_t0_y1 + f_t0_y2, f_t0_y12, rtol=1e-10, atol=1e-12):
            return _fallback_solve(f, t_span, y0, deadline)
            
        # Check homogeneity: f(c*y) == c*f(y)
        c = 2.0
        f_t0_cy1 = f(t0, c * y_test1)
        if not np.allclose(c * f_t0_y1, f_t0_cy1, rtol=1e-10, atol=1e-12):
            return _fallback_solve(f, t_span, y0, deadline)
            
        # Check time-invariance: f(t0, y) == f(t1, y)
        # Use a point in the middle to avoid boundary effects if any
        t_mid = 0.5 * (t0 + t1)
        f_tmid_y1 = f(t_mid, y_test1)
        if not np.allclose(f_t0_y1, f_tmid_y1, rtol=1e-10, atol=1e-12):
            # Time-dependent. For heat equation, this is unlikely, but handle it.
            # We could try to extract A(t) and use a product of exponentials, 
            # but that's complex. Fallback to Radau which handles time-dependent stiff ODEs well.
            return _fallback_solve(f, t_span, y0, deadline)
            
    except Exception:
        return _fallback_solve(f, t_span, y0, deadline)

    # 2. Extract the Linear Operator A
    # f(t, y) = A @ y
    A = np.empty((n, n))
    for j in range(n):
        if timed_out():
            # If we run out of time during extraction, we can't finish the expm method.
            # We have partial A, but it's useless. Fallback is too late if we've spent most time here.
            # However, 100 calls is very fast. This is unlikely.
            return _fallback_solve(f, t_span, y0, deadline)
            
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)

    # 3. Compute the Solution using Matrix Exponential
    dt = t1 - t0
    try:
        # scipy.linalg.expm is robust for small/medium matrices
        # Use 'slogdet' or other options? No, just standard expm.
        # For 10x10, this is extremely fast and accurate.
        M = scipy.linalg.expm(A * dt)
        y_end = M @ y0
    except Exception:
        # Fallback if expm fails
        return _fallback_solve(f, t_span, y0, deadline)

    return y_end

def _fallback_solve(f, t_span, y0, deadline):
    """
    Fallback solver for non-linear or time-dependent cases.
    Uses scipy.integrate.solve_ivp with Radau (implicit, good for stiff) 
    and tight tolerances to maximize accuracy score.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    try:
        # Radau is an implicit Runge-Kutta method, good for stiff problems.
        # It requires Jacobian, which solve_ivp will approximate.
        # We use tight tolerances to get high accuracy (low error term in score).
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-12, 
            atol=1e-12,
            max_step=(t1-t0)/100 # Ensure some steps, though adaptive will handle it
        )
        
        if sol.success and time.time() < deadline:
            return sol.y[:, -1]
            
        # If Radau fails or is slow, try DOP853 (explicit, high order)
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-12, 
            atol=1e-12
        )
        
        if sol.success:
            return sol.y[:, -1]
            
    except Exception:
        pass
        
    # Last resort: return initial state if everything fails (score will be 0, but better than exception)
    return y0
