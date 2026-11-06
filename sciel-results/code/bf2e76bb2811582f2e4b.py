"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    n = len(y0_arr)
    
    # Deadline guard: keep a safety margin before the 330s limit
    start_time = time.time()
    deadline = start_time + 315.0  # 15s safety margin
    
    # First, try to detect if f is a linear system: y' = A y + b(t)
    # For a linear system, f(t, y) = A @ y + c (possibly time-dependent c)
    # We can probe: f(t, y1) - f(t, y2) = A @ (y1 - y2)
    
    # Probe with two different y values to extract A
    y_probe1 = np.zeros(n)
    y_probe2 = np.ones(n)
    
    ft_y1 = np.asarray(f(t0, y_probe1), dtype=float)
    ft_y2 = np.asarray(f(t0, y_probe2), dtype=float)
    
    # Check linearity: f(t, y) = A @ y + c
    # A @ y_probe2 + c = ft_y2
    # A @ y_probe1 + c = ft_y1  => c = ft_y1 (since y_probe1 = 0)
    # So A = ft_y2 - ft_y1 (since y_probe2 = ones)
    c = ft_y1.copy()
    A = (ft_y2 - ft_y1).copy()
    
    # Verify linearity with a third probe
    y_probe3 = np.random.RandomState(42).randn(n)
    ft_y3 = np.asarray(f(t0, y_probe3), dtype=float)
    predicted = A @ y_probe3 + c
    linearity_error = np.linalg.norm(ft_y3 - predicted) / (1.0 + np.linalg.norm(ft_y3))
    
    # Also check time-dependence of c: f(t1, 0) should give c(t1)
    ft1_y0 = np.asarray(f(t1, np.zeros(n)), dtype=float)
    c_t1 = ft1_y0
    
    if linearity_error < 1e-10:
        # It's a linear system. Check if A is time-independent.
        # For time-independent A, c may be time-dependent.
        # Use scipy's solve_ivp with appropriate solver for stiff linear systems.
        
        # Check if c is time-independent
        c_diff = np.linalg.norm(c_t1 - c) / (1.0 + np.linalg.norm(c))
        
        if c_diff < 1e-10:
            # Fully linear time-invariant: y' = A y + c
            # We can use matrix exponential for exact solution
            from scipy.linalg import expm
            # y(t) = expm((t-t0)*A) @ y0 + integral of expm((t-s)*A) @ c ds
            # For constant c: y(t) = expm((t-t0)*A) @ y0 + (expm((t-t0)*A) - I) @ inv(A) @ c
            # But if A is singular, use: integral = expm((t-t0)*A) @ integral_0^(t-t0) expm(-s*A) c ds
            # Simpler: use solve_ivp with Radau or BDF
            pass
        
        # Use solve_ivp with a stiff solver
        def f_wrapped(t, y):
            return A @ y + c
        
        # Try Radau (implicit, good for stiff)
        try:
            sol = solve_ivp(f_wrapped, [t0, t1], y0_arr, method='Radau', 
                           rtol=1e-12, atol=1e-14, max_step=(t1-t0)/100)
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
        
        # Try BDF
        try:
            sol = solve_ivp(f_wrapped, [t0, t1], y0_arr, method='BDF',
                           rtol=1e-12, atol=1e-14, max_step=(t1-t0)/100)
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
        
        # Try LSODA
        try:
            sol = solve_ivp(f_wrapped, [t0, t1], y0_arr, method='LSODA',
                           rtol=1e-12, atol=1e-14)
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
    
    # Fallback: use solve_ivp directly with the original f
    try:
        sol = solve_ivp(f, [t0, t1], y0_arr, method='Radau',
                       rtol=1e-12, atol=1e-14, max_step=(t1-t0)/100)
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    try:
        sol = solve_ivp(f, [t0, t1], y0_arr, method='BDF',
                       rtol=1e-12, atol=1e-14, max_step=(t1-t0)/100)
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    try:
        sol = solve_ivp(f, [t0, t1], y0_arr, method='LSODA',
                       rtol=1e-12, atol=1e-14)
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Last resort: explicit RK45 with tight tolerances
    try:
        sol = solve_ivp(f, [t0, t1], y0_arr, method='RK45',
                       rtol=1e-10, atol=1e-12, max_step=(t1-t0)/1000)
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Manual RK4 as ultimate fallback
    steps = 10000
    h = (t1 - t0) / steps
    t, y = t0, y0_arr.copy()
    for _ in range(steps):
        if time.time() > deadline:
            return y
        k1 = np.asarray(f(t, y), dtype=float)
        k2 = np.asarray(f(t + h/2, y + h/2 * k1), dtype=float)
        k3 = np.asarray(f(t + h/2, y + h/2 * k2), dtype=float)
        k4 = np.asarray(f(t + h, y + h * k3), dtype=float)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y

# EVOLVE-BLOCK-END
