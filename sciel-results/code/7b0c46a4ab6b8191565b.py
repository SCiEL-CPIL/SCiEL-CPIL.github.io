"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra system y0' = 1.5 y0 - y0 y1, y1' = -3 y1 + y0 y1
    from t_span[0] to t_span[1] with initial condition y0.
    
    Uses DOP853 (8th-order Dormand-Prince) with very tight tolerances to match
    the reference solution closely while minimizing function calls.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin of 20 seconds
    
    # Use scipy's solve_ivp with DOP853 and very tight tolerances
    # This matches the reference method exactly
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y,
            method='DOP853',
            rtol=1e-11,
            atol=1e-13
        )
        
        if sol.success and sol.t[-1] == t1:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)) and y_end.shape == y0.shape:
                return y_end
    except Exception:
        pass
    
    # Fallback: implement DOP853 manually if scipy fails
    # DOP853 coefficients (8th order, 5th order error estimate, 3rd order embedded)
    # This is a complex method, so let's use a simpler high-order approach as fallback
    
    def rk4_step(t, y, h):
        """Single RK4 step."""
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2*k1)
        k3 = f(t + h/2, y + h/2*k2)
        k4 = f(t + h, y + h*k3)
        return y + h/6*(k1 + 2*k2 + 2*k3 + k4)
    
    # Adaptive RK4 with error estimation
    h = (t1 - t0) / 5000
    t = t0
    max_steps = 500000
    min_h = (t1 - t0) / max_steps
    step = 0
    
    while t < t1 - min_h/10:
        if time.time() > deadline:
            break
        
        if t + h > t1:
            h = t1 - t
        
        # Error estimation: compare two half-steps vs one full step
        y_half1 = rk4_step(t, y, h/2)
        y_half2 = rk4_step(t + h/2, y_half1, h/2)
        y_full = rk4_step(t, y, h)
        
        err = np.linalg.norm(y_half2 - y_full) / (np.linalg.norm(y_half2) + 1e-30)
        
        tol = 1e-11 * (1 + np.linalg.norm(y))
        if err < tol:
            y = y_half2  # Use the more accurate half-step result
            t += h
            if err < tol / 10:
                h = min(h * 1.4, (t1 - t) / 5)
        else:
            h = max(h * 0.5 * (tol / err) ** 0.25, min_h)
        
        step += 1
        if step > max_steps:
            break
    
    # Take remaining steps
    while t < t1:
        if time.time() > deadline:
            break
        h_rem = min(h, t1 - t)
        y = rk4_step(t, y, h_rem)
        t += h_rem
    
    if not np.all(np.isfinite(y)):
        # Last resort: fixed small steps
        n_steps = 10000
        h = (t1 - t0) / n_steps
        y = np.array(y0, dtype=float)
        t = t0
        for _ in range(n_steps):
            y = rk4_step(t, y, h)
            t += h
    
    return y

# EVOLVE-BLOCK-END
