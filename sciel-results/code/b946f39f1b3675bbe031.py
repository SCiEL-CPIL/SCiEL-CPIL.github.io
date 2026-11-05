"""Rewritten program using Implicit Crank-Nicolson with Linear Operator Extraction."""
import numpy as np
import scipy.linalg as la

def _rk4(f, t0, t1, y0, steps):
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            return None
    return y

def _is_linear(f, t0, y0, A, g, tol=1e-9):
    """Check if f(t, y) is approximately A @ y + g."""
    # Check at y0
    y_actual = f(t0, y0)
    y_model = A @ y0 + g
    norm_actual = np.linalg.norm(y_actual)
    if norm_actual < 1e-12:
        return np.linalg.norm(y_model) < 1e-9
    rel_err = np.linalg.norm(y_actual - y_model) / norm_actual
    return rel_err < tol

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    N = len(y0_arr)
    
    # 1. Probe for Linear Operator A and Constant g
    # Assume f(t, y) = A @ y + g
    A = np.zeros((N, N))
    e = np.zeros(N)
    
    # Probe columns of A
    for j in range(N):
        e[j] = 1.0
        v = f(t0, e)
        A[:, j] = v
        e[j] = 0.0
        
    # Probe constant term g
    g = f(t0, np.zeros(N))
    
    # 2. Verify Linearity
    # Check if the model holds for the initial condition
    if not _is_linear(f, t0, y0_arr, A, g):
        # Fallback to explicit RK4 if not linear
        return _rk4(f, t0, t1, y0_arr, 2000)
        
    # 3. Crank-Nicolson Integration
    # (I - h/2 A) y_{n+1} = (I + h/2 A) y_n + h g
    steps = 10000
    h = (t1 - t0) / steps
    
    I = np.eye(N)
    M_left = I - 0.5 * h * A
    M_right = I + 0.5 * h * A
    
    # Factorize M_left once
    # Using LU factorization for dense matrix
    lu_piv = la.lu_factor(M_left)
    
    y_curr = y0_arr.copy()
    
    for _ in range(steps):
        # RHS = M_right @ y_curr + h * g
        rhs = M_right @ y_curr + h * g
        # Solve M_left @ y_next = rhs
        y_next = la.lu_solve(lu_piv, rhs)
        y_curr = y_next
        
    return y_curr
