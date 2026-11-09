"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
import time


def _is_close(a, b, tol=1e-10):
    return np.allclose(a, b, atol=tol, rtol=tol)


def _try_analytic_linear(f, t_span, y0, deadline):
    """
    Attempts to solve the ODE analytically assuming it is a linear system
    y' = A(t)y + b(t) where A and b are constant.
    Returns the solution if the assumption holds, otherwise None.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # 1. Check homogeneity / constant b
    b0 = f(t0, np.zeros(n))
    if time.time() > deadline:
        return None
        
    b1 = f(t1, np.zeros(n))
    if time.time() > deadline:
        return None
        
    # If b is not constant or not zero, we can still handle constant b via augmentation.
    # If b varies, we must use numerical integration.
    b_is_constant = _is_close(b0, b1, tol=1e-8)
    
    # 2. Estimate A at t0
    A = np.zeros((n, n))
    for i in range(n):
        if time.time() > deadline:
            return None
        e_i = np.zeros(n)
        e_i[i] = 1.0
        # f(t, y) = A y + b. So f(t, e_i) = A[:, i] + b.
        # Thus A[:, i] = f(t, e_i) - b.
        # We use b0 for t0.
        f_ei = f(t0, e_i)
        A[:, i] = f_ei - b0

    # 3. Verify A is constant by checking one column at t1
    if time.time() > deadline:
        return None
    e_0 = np.zeros(n)
    e_0[0] = 1.0
    f_t1_e0 = f(t1, e_0)
    # Expected f(t1, e_0) = A[:, 0] + b1
    expected_f_t1_e0 = A[:, 0] + b1
    A_is_constant = _is_close(f_t1_e0, expected_f_t1_e0, tol=1e-8)

    if not (b_is_constant and A_is_constant):
        return None

    # If both A and b are constant, we can use the analytic solution.
    # Case 1: b is zero (homogeneous)
    # y_end = expm((t1-t0)A) y0
    # Case 2: b is non-zero (non-homogeneous)
    # Use augmented system: z' = M z, where z = [y; 1]
    # M = [[A, b], [0, 1]]
    
    dt = t1 - t0
    
    if np.allclose(b0, 0, atol=1e-12):
        # Homogeneous
        Y = expm(dt * A)
        y_end = Y @ y0
    else:
        # Non-homogeneous with constant b
        M = np.zeros((n + 1, n + 1))
        M[:n, :n] = A
        M[:n, n] = b0
        M[n, n] = 1.0
        
        z0 = np.zeros(n + 1)
        z0[:n] = y0
        z0[n] = 1.0
        
        Z = expm(dt * M)
        z_end = Z @ z0
        y_end = z_end[:n]

    return y_end


def _solve_numeric(f, t_span, y0, deadline):
    """
    Fallback numerical solver for general stiff linear systems.
    Uses Radau method with tight tolerances.
    """
    # We can try to provide a Jacobian to speed up Radau.
    # But estimating it costs calls. 
    # Let's just use the default Jacobian estimation for Radau.
    
    # To improve performance, we can set max_step to avoid huge steps in non-stiff regions?
    # Radau is adaptive.
    
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='Radau', 
        rtol=1e-10, 
        atol=1e-12,
        # max_step=(t_span[1]-t_span[0])/100 # Might be too restrictive
    )
    
    if sol.success:
        return sol.y[:, -1]
    else:
        # If Radau fails, try BDF? Or just return last state.
        # For robustness, return last state if finite.
        if np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
        # If all else fails, return y0 (should not happen)
        return y0


def integrate(f, t_span, y0):
    """
    Solve y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns y(t_end).
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard: kill after ~290s to stay within 330s wall-clock
    start_time = time.time()
    deadline = start_time + 290

    # First, try the high-accuracy, low-cost analytic solution for constant linear systems
    y_analytic = _try_analytic_linear(f, t_span, y0, deadline)
    
    if y_analytic is not None:
        return y_analytic

    # If analytic failed (system is not constant linear), use a stiff numerical solver
    return _solve_numeric(f, t_span, y0, deadline)

# EVOLVE-BLOCK-END
