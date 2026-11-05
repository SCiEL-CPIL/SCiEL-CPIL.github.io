import numpy as np
import scipy.linalg

def _check_linearity(f, t, y0):
    """
    Checks if f is linear in y.
    Returns (is_linear, A) where A is the matrix such that f(t, y) = A @ y.
    """
    n = len(y0)
    # Test f(t, 0) == 0
    y_zero = np.zeros(n)
    f_zero = f(t, y_zero)
    if np.linalg.norm(f_zero) > 1e-10:
        return False, None

    # Extract matrix A using basis vectors
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(t, e_j)

    # Verify linearity with random vectors
    np.random.seed(42)
    y1 = np.random.randn(n)
    y2 = np.random.randn(n)
    
    f_sum = f(t, y1 + y2)
    f_ind = f(t, y1) + f(t, y2)
    
    # Check A @ (y1+y2) vs f_sum
    A_sum = A @ (y1 + y2)
    
    err1 = np.linalg.norm(f_sum - f_ind)
    err2 = np.linalg.norm(f_sum - A_sum)
    
    scale = max(1.0, np.linalg.norm(f_sum))
    
    if err1 > 1e-8 * scale or err2 > 1e-8 * scale:
        return False, None
        
    return True, A

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    is_linear, A = _check_linearity(f, t0, y0)
    
    if is_linear:
        # Use matrix exponential for exact solution of linear system
        # y(t1) = exp(A * (t1 - t0)) @ y0
        dt = t1 - t0
        M = scipy.linalg.expm(A * dt)
        y_end = M @ y0
        return y_end
    else:
        # Fallback: High-order implicit method or RK4 with many steps
        # Since we expect linearity, this is just a safety net.
        # Use RK4 with a large number of steps for accuracy
        steps = 10000
        h = (t1 - t0) / steps
        t, y = t0, y0.copy()
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        return y
