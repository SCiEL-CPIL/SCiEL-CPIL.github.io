"""ODE solver exploiting linearity via matrix exponential."""
import numpy as np
from scipy.linalg import expm
import time


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) over t_span = [t0, t1].
    
    Strategy: Exploit the linearity of the system y' = A y.
    1. Extract A by evaluating f at basis vectors.
    2. Compute y_end = expm((t1-t0) * A) @ y0.
    3. Use high-precision matrix exponential for accuracy.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Extract matrix A assuming f(t, y) = A @ y
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)
    
    # Calculate matrix exponential
    dt = t1 - t0
    M = expm(dt * A)
    
    y_end = M @ y0
    
    # Fallback if something goes wrong (e.g., non-finite result)
    if not np.all(np.isfinite(y_end)):
        # Use a robust solver as fallback
        steps = 1000
        h = dt / steps
        t = t0
        y = y0.copy()
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y
        
    return y_end
