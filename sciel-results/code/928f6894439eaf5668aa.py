import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # 1. Probe for linearity and time-invariance
    # We will check if f(t, y) = A y for some constant A.
    
    # Strategy:
    # 1. Estimate A using n calls at t0.
    # 2. Verify A by checking f(t1, y0) against A @ y0.
    # 3. If verification fails, fall back to solve_ivp.
    
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)
        
    # Verify time-invariance and linearity consistency
    # Check f(t1, y0) vs A @ y0
    f_t1_y0 = f(t1, y0)
    pred_t1_y0 = A @ y0
    
    # Also check a random point? Maybe not needed if t1 check passes.
    # Tolerance for verification
    err = np.linalg.norm(f_t1_y0 - pred_t1_y0)
    scale = max(1.0, np.linalg.norm(f_t1_y0))
    
    if err / scale < 1e-10:
        # Assume y' = A y
        # Solution is expm(A * (t1 - t0)) @ y0
        dt = t1 - t0
        M = A * dt
        # For large M, expm might be unstable? No, expm handles scaling.
        # However, if A has large eigenvalues, expm is fine.
        expM = expm(M)
        y_end = expM @ y0
        return y_end
    else:
        # Fall back to robust integrator
        # Use Radau for stiffness
        # Provide Jacobian if we think it's linear but time-dependent? 
        # If it's linear but time-dependent, A(t) varies. 
        # We can't easily provide a constant Jacobian.
        # But if it's linear and time-invariant, A is constant.
        # If verification failed, maybe it's non-linear or time-dependent.
        # However, if it's linear and time-invariant, the failure is numerical.
        # Providing A as Jacobian is a good guess.
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-12, 
            atol=1e-14,
            jac=lambda t, y: A
        )
        return sol.y[:, -1]
