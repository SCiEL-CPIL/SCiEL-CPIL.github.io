"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1].
    
    Strategy:
    1. Try to detect if the system is linear constant-coefficient (f(t,y) = A@y + b).
       If so, use exact matrix exponential solution.
    2. Otherwise, use a robust adaptive stiff solver.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Check if system is linear: f(t, y) = A @ y + b
    # Probe at two different times with zero and non-zero y to detect linearity
    # and time-independence
    
    # First, get f at t0 with y0
    f0 = f(t0, y0)
    
    # Get f at t1 with y0 to check time dependence
    f1 = f(t1, y0)
    
    # Check if f is time-invariant (constant coefficient)
    # For linear systems, f(t, y) = A@y + b, so f(t, 0) = b for all t
    f_t0_zero = f(t0, np.zeros(n))
    f_t1_zero = f(t1, np.zeros(n))
    
    # Check if b is constant (time-invariant)
    is_time_invariant = np.allclose(f_t0_zero, f_t1_zero, rtol=1e-12, atol=1e-12)
    
    if is_time_invariant:
        # Try to extract A and b assuming f(t, y) = A @ y + b
        b = f_t0_zero
        
        # Extract A by probing with basis vectors
        # A @ y + b - b = A @ y
        # So A @ y = f(t, y) - b
        # Probe with standard basis vectors
        
        A = np.zeros((n, n))
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            f_ej = f(t0, e_j)
            A[:, j] = f_ej - b
        
        # Verify linearity: check that f(t, y) = A @ y + b for a few random test points
        is_linear = True
        for _ in range(5):
            y_test = np.random.randn(n) * 0.5
            f_test = f(t0, y_test)
            f_pred = A @ y_test + b
            if not np.allclose(f_test, f_pred, rtol=1e-10, atol=1e-10):
                is_linear = False
                break
        
        if is_linear:
            # Use exact matrix exponential solution
            # y(t) = e^{A(t-t0)} @ y0 + integral of e^{A(t-s)} b ds from t0 to t
            # For constant b: y(t) = e^{A(T)} @ y0 + (e^{A(T)} - I) @ A^{-1} @ b (if A invertible)
            # More robust: use scipy.linalg.expm
            try:
                from scipy.linalg import expm
                T = t1 - t0
                exp_AT = expm(A * T)
                
                # Compute particular solution for constant b
                # If A is invertible: (exp_AT - I) @ inv(A) @ b
                # If A is singular, use numerical integration or series
                try:
                    # Try direct inverse
                    A_inv = np.linalg.inv(A)
                    y_particular = (exp_AT - np.eye(n)) @ A_inv @ b
                except np.linalg.LinAlgError:
                    # A is singular; use series expansion or numerical quadrature
                    # y_particular = integral_0^T e^{A*s} b ds
                    # Use Gauss-Legendre quadrature with enough points
                    m = 100
                    xs, ws = np.polynomial.legendre.leggauss(m)
                    # Map from [-1, 1] to [0, T]
                    s_vals = 0.5 * T * (xs + 1)
                    w_vals = 0.5 * T * ws
                    y_particular = np.zeros(n)
                    for s, w in zip(s_vals, w_vals):
                        exp_As = expm(A * s)
                        y_particular += w * (exp_As @ b)
                
                y_final = exp_AT @ y0 + y_particular
                return y_final
            except Exception:
                pass
    
    # Fallback: use a stiff solver
    # Use Radau or BDF from scipy if available
    try:
        from scipy.integrate import solve_ivp
        
        def f_wrapper(t, y):
            return f(t, y)
        
        sol = solve_ivp(f_wrapper, [t0, t1], y0, method='Radau', rtol=1e-12, atol=1e-14, max_step=0.01)
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback to BDF
            sol = solve_ivp(f_wrapper, [t0, t1], y0, method='BDF', rtol=1e-12, atol=1e-14, max_step=0.01)
            if sol.success:
                return sol.y[:, -1]
    except Exception:
        pass
    
    # Last resort: fixed-step implicit Euler with many steps
    h = (t1 - t0) / 10000
    t, y = t0, y0.copy()
    n_steps = 10000
    for _ in range(n_steps):
        # Simple implicit Euler: y_new = y + h * f(t+h, y_new)
        # Use fixed-point iteration (3 iterations)
        z = y + h * f(t + h, y)
        for _ in range(3):
            z = y + h * f(t + h, z)
        y = z
        t += h
    return y
# EVOLVE-BLOCK-END
