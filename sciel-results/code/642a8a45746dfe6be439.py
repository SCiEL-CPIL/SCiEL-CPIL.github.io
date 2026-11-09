"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time


def _is_linear_autonomous(f, y0, t0, t1, tol=1e-10):
    """
    Check if f is autonomous (independent of t) and linear in y.
    Returns True if checks pass, False otherwise.
    """
    n = len(y0)
    y0 = np.array(y0, dtype=float)
    
    # Check autonomy: f(t0, y0) vs f(t1, y0)
    f_t0 = f(t0, y0)
    f_t1 = f(t1, y0)
    if np.linalg.norm(f_t0 - f_t1) > tol * max(1.0, np.linalg.norm(f_t0)):
        return False
    
    # Check linearity: f(0, 2*y0) == 2*f(0, y0) and f(0, y0 + e) == f(0, y0) + f(0, e)
    f_2y = f(t0, 2.0 * y0)
    if np.linalg.norm(f_2y - 2.0 * f_t0) > tol * max(1.0, np.linalg.norm(f_t0)):
        return False
    
    # Check additivity with a random vector
    e = np.ones(n) * 1e-5
    f_sum = f(t0, y0 + e)
    f_e = f(t0, e)
    if np.linalg.norm(f_sum - (f_t0 + f_e)) > tol * max(1.0, np.linalg.norm(f_t0), np.linalg.norm(f_e)):
        return False
        
    return True


def _solve_linear_autonomous(f, t_span, y0):
    """
    Solve y' = A y by computing expm(A * dt) @ y0.
    This requires n calls to f to extract A.
    """
    t0, t1 = t_span
    dt = t1 - t0
    n = len(y0)
    y0 = np.array(y0, dtype=float)
    
    # Extract A
    A = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        A[:, j] = f(t0, e)
        
    # Compute matrix exponential
    M = expm(A * dt)
    y_end = M @ y0
    
    return y_end


def _rk45_adaptive(f, t_span, y0, rtol=1e-12, atol=1e-14, max_steps=100000):
    """
    High-order adaptive Runge-Kutta method (Fehlberg 7(8) or similar via scipy if available, 
    but let's stick to a robust RK4 with tight tolerance or just use solve_ivp logic manually).
    Actually, let's use a simple but accurate RK4 with many steps or an implicit method if needed.
    Given the linear case is handled above, this is a fallback.
    """
    t0, t1 = t_span
    h = (t1 - t0) / 10000
    t, y = t0, np.array(y0, dtype=float)
    steps = 10000
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y


def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Try to detect linear autonomous system
    # We need to be careful with the number of calls.
    # _is_linear_autonomous uses ~4 calls.
    # _solve_linear_autonomous uses n calls.
    # Total ~ n + 4 calls.
    
    try:
        if _is_linear_autonomous(f, y0, t0, t1):
            if time.time() < deadline:
                y_end = _solve_linear_autonomous(f, t_span, y0)
                if np.all(np.isfinite(y_end)):
                    return y_end
    except Exception:
        pass
    
    # Fallback: RK4 with fine steps
    y_end = _rk45_adaptive(f, t_span, y0)
    return y_end
# EVOLVE-BLOCK-END
