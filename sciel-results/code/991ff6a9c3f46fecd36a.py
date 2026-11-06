import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280  # 20s margin from 300s limit
    
    # 1. Attempt Linear Constant Coefficient Detection
    # Probe b at t0
    b = f(t0, np.zeros(n))
    
    # Check time invariance of b at t1
    b_t1 = f(t1, np.zeros(n))
    if not np.allclose(b, b_t1, rtol=1e-12, atol=1e-12):
        # Not time invariant. Fall back to solver.
        return _fallback_solve(f, t0, t1, y0, start_time, deadline)
        
    # Extract A using basis vectors
    # A[:, j] = f(t0, e_j) - b
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j) - b
        
    # Verify linearity with deterministic test vectors
    # We assume the system is linear based on the task description, 
    # but we do a quick check to be safe.
    is_linear = True
    test_vecs = [
        np.ones(n),
        np.arange(1, n+1, dtype=float),
        np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n)])
    ]
    for y_test in test_vecs:
        f_test = f(t0, y_test)
        f_pred = A @ y_test + b
        if not np.allclose(f_test, f_pred, rtol=1e-10, atol=1e-10):
            is_linear = False
            break
            
    if is_linear:
        if time.time() > deadline:
             return y0 # Should not happen, but safety
        
        try:
            from scipy.linalg import expm
            T = t1 - t0
            exp_AT = expm(A * T)
            
            # Compute particular solution integral
            # y_p(T) = integral_0^T exp(A*u) b du
            # If A is invertible: (exp(AT) - I) A^{-1} b
            # If A is singular: use quadrature or other methods
            
            try:
                # Try invertible case
                # Use solve instead of inv for better numerical stability
                # x = A^{-1} b  =>  A x = b
                A_inv_b = np.linalg.solve(A, b)
                y_particular = (exp_AT - np.eye(n)) @ A_inv_b
            except np.linalg.LinAlgError:
                # Singular case: Use Gauss-Legendre quadrature
                # For a stiff system, we might need more points, but 100 is usually enough for smooth expm
                m = 100
                xs, ws = np.polynomial.legendre.leggauss(m)
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
            
    # 2. Fallback Solver
    return _fallback_solve(f, t0, t1, y0, start_time, deadline)

def _fallback_solve(f, t0, t1, y0, start_time, deadline):
    # Use solve_ivp with Radau for stiffness
    try:
        from scipy.integrate import solve_ivp
        # Use a wrapper to ensure we don't exceed deadline? 
        # solve_ivp doesn't easily support time limits.
        # We'll just run it. If it's too slow, it will time out the whole program, 
        # but this is a fallback for non-linear cases which shouldn't happen per task description.
        sol = solve_ivp(f, [t0, t1], y0, method='Radau', rtol=1e-10, atol=1e-12, max_step=0.01)
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
        
    # Last resort: Fixed step implicit Euler
    # This is likely to score poorly due to call count, but better than crashing
    n_steps = 1000
    h = (t1 - t0) / n_steps
    y = y0.copy()
    t = t0
    for _ in range(n_steps):
        z = y
        for _ in range(3):
            z = y + h * f(t + h, z)
        y = z
        t += h
    return y
