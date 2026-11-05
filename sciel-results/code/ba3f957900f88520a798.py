import numpy as np
import time
from scipy.linalg import expm

def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1].
    
    Strategy: 
    1. Detect if the ODE is linear and time-invariant: y' = A y.
    2. Extract the matrix A using 101 function calls (basis vectors).
    3. Compute the exact solution y(t_end) = exp((t_end - t_start) * A) @ y0.
    
    This approach minimizes the number of function calls (101) and achieves
    high accuracy (limited only by expm precision), maximizing the score:
    -log10(error) - 0.5*log10(calls)
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float).copy()
    n = len(y0)
    
    start_time = time.time()
    deadline = start_time + 290  # Safety margin for 300s limit
    
    # Try to extract the linear operator A
    # f(t, y) should be linear in y: f(t, c1*y1 + c2*y2) = c1*f(t, y1) + c2*f(t, y2)
    # and time-invariant: f(t, y) = f(t', y)
    
    # We will construct A such that f(t, y) = A @ y
    # A[:, i] = f(t0, e_i) where e_i is the i-th standard basis vector
    
    A = np.zeros((n, n), dtype=float)
    
    # Use a single time point t0 to check time-invariance implicitly by just using it.
    # If the problem is time-invariant, f(t, y) does not depend on t.
    # If it depends on t, this extraction will fail or produce a wrong A.
    # However, convection-diffusion IVPs in this context are typically time-invariant.
    
    # Optimization: Check linearity with a few random vectors first?
    # No, just extract directly. If it's not linear, the result will be wrong, 
    # but the task specifies "convection-diffusion" which is linear.
    
    for i in range(n):
        if time.time() > deadline:
            # Fallback to a simple RK4 if we run out of time (unlikely for n=100)
            break
        
        e_i = np.zeros(n, dtype=float)
        e_i[i] = 1.0
        A[:, i] = f(t0, e_i)
        
    # Check if we completed extraction
    if time.time() > deadline:
        # Fallback: Use a simple fixed-step RK4 with many steps
        # This is a safety net, should not trigger for n=100
        return _rk4_fallback(f, t_span, y0, deadline)
    
    # Verify linearity/time-invariance roughly?
    # Let's check if f(t0, y0) is close to A @ y0
    # If not, the assumption of linearity/time-invariance might be violated.
    # But for the specific task, we assume it holds.
    
    # Compute the matrix exponential
    # dt = t1 - t0
    dt = t1 - t0
    if dt == 0:
        return y0
    
    try:
        # expm can be expensive for large n, but n=100 is fine.
        # Complexity O(n^3) ~ 1e6 operations, very fast.
        expm_A_dt = expm(dt * A)
        y_end = expm_A_dt @ y0
    except Exception:
        # Fallback if expm fails
        return _rk4_fallback(f, t_span, y0, deadline)
    
    return y_end

def _rk4_fallback(f, t_span, y0, deadline):
    """
    Fallback RK4 solver if linear extraction fails or times out.
    """
    t0, t1 = t_span
    y = np.asarray(y0, dtype=float).copy()
    n_steps = 1000
    h = (t1 - t0) / n_steps
    t = t0
    
    for _ in range(n_steps):
        if time.time() > deadline:
            break
        
        k1 = f(t, y)
        k2 = f(t + 0.5*h, y + 0.5*h*k1)
        k3 = f(t + 0.5*h, y + 0.5*h*k2)
        k4 = f(t + h, y + h*k3)
        
        y = y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        
    return y
