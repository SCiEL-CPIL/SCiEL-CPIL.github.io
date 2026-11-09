"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time


def _build_matrix(f, t0, y0):
    """Probe the system to determine if it's linear: y' = A*y + b.
    
    Returns (A, b, is_linear) where:
    - A is the coefficient matrix (n x n)
    - b is the constant vector (n,)
    - is_linear is True if the system appears linear with constant coefficients
    """
    n = len(y0)
    
    # Evaluate b = f(t0, 0)
    y_zero = np.zeros(n)
    b = np.array(f(t0, y_zero), dtype=float)
    
    # Evaluate A columns: A[:, j] = f(t0, e_j)
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = np.array(f(t0, e_j), dtype=float)
    
    # Check if the system is consistent with y' = A*y + b at a different point
    # Use a random test point to verify linearity and constant coefficients
    y_test = np.ones(n) * 0.5
    predicted = A @ y_test + b
    actual = np.array(f(t0, y_test), dtype=float)
    
    rel_err = np.linalg.norm(predicted - actual) / max(1.0, np.linalg.norm(actual))
    is_linear = rel_err < 1e-10
    
    return A, b, is_linear


def _arnoldi_exp_action(A, b, t0, t1, y0, max_k=100):
    """Use Arnoldi iteration to compute exp(A*dt)*y0 + integral term.
    
    This avoids explicitly forming the full matrix exponential.
    Uses Krylov subspace projection.
    """
    n = len(y0)
    dt = t1 - t0
    
    # Build augmented system: z = [y; 1], z' = M*z
    # M = [[A, b], [0, 0]]
    # We want exp(M*dt) * z0 where z0 = [y0; 1]
    
    # Define the linear operator
    def M_op(v):
        """Apply M to v (v has length n+1)."""
        result = np.empty_like(v)
        result[:n] = A @ v[:n] + v[n] * b
        result[n] = 0.0
        return result
    
    # Initial vector
    z0 = np.zeros(n + 1)
    z0[:n] = y0
    z0[n] = 1.0
    
    # Arnoldi iteration to build orthonormal basis V and Hessenberg H
    m = min(max_k, n + 1)
    V = np.zeros((n + 1, m + 1))
    H = np.zeros((m + 1, m))
    
    v0 = z0 / (np.linalg.norm(z0) + 1e-30)
    V[:, 0] = v0
    
    for j in range(m):
        w = M_op(V[:, j])
        
        # Modified Gram-Schmidt
        for i in range(j + 1):
            H[i, j] = V[:, i] @ w
            w -= H[i, j] * V[:, i]
        
        H[j + 1, j] = np.linalg.norm(w)
        
        if H[j + 1, j] < 1e-14:
            m = j + 1
            break
        
        V[:, j + 1] = w / H[j + 1, j]
    
    # Now we have V (n+1 x m) and H (m x m) such that M*V[:, :m] ≈ V[:, :m] @ H
    # We want exp(M*dt) * z0 = V @ (exp(H*dt) * (V^T * z0))
    
    # V^T * z0
    Vt_z0 = V[:, :m] @ z0
    
    # Compute exp(H*dt)
    H_dt = H[:m, :m] * dt
    expH_dt = expm(H_dt)
    
    # Result
    result = V[:, :m] @ (expH_dt @ Vt_z0)
    
    return result[:n]


def _solve_linear_constant(A, b, t0, t1, y0):
    """Solve y' = A*y + b with constant A, b using matrix exponential.
    
    The solution is:
    y(t) = expm(A*(t-t0)) @ y0 + integral from t0 to t of expm(A*(s-t0)) @ b ds
    
    For constant b, the integral can be computed as:
    integral = expm(A*(t-t0)) @ A^{-1} @ (expm(A*(t-t0)) - I) @ b  (if A is invertible)
    Or more robustly, use the augmented matrix method.
    """
    n = len(y0)
    dt = t1 - t0
    
    # Method 1: Use augmented matrix for non-homogeneous case
    # Consider z = [y; 1], then z' = [[A, b], [0, 0]] @ z
    # z(0) = [y0; 1]
    # z(dt) = expm([[A, b], [0, 0]] * dt) @ [y0; 1]
    
    # Build augmented matrix
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = A
    M[:n, n] = b
    # Last row is all zeros
    
    z0 = np.zeros(n + 1)
    z0[:n] = y0
    z0[n] = 1.0
    
    # Compute matrix exponential
    M_dt = M * dt
    expM = expm(M_dt)
    
    z_end = expM @ z0
    
    return z_end[:n]


def _solve_linear_nonconstant(f, t0, t1, y0, max_steps=10000):
    """Solve a linear system y' = A(t)*y + b(t) using adaptive step sizes.
    
    For this task, we assume the system is actually constant-coefficient,
    so this is a fallback.
    """
    # This is a simple explicit RK4 as fallback
    steps = min(max_steps, 5000)
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    n = len(y)
    
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


def integrate(f, t_span, y0):
    """Integrate y' = f(t, y) from t_span[0] to t_span[1].
    
    Uses the structure of the 15-dimensional stiff linear system:
    1. Probe to determine if the system is linear with constant coefficients
    2. If so, use matrix exponential for exact solution
    3. Otherwise, fall back to adaptive methods
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 300  # 300 seconds max
    
    # Step 1: Probe the system to see if it's linear with constant coefficients
    A, b, is_linear = _build_matrix(f, t0, y0)
    
    # Check time
    if time.time() > deadline - 5:
        # Not enough time, use a quick RK4
        return _solve_linear_nonconstant(f, t0, t1, y0, max_steps=1000)
    
    if is_linear:
        # Use matrix exponential for exact solution
        try:
            y_end = _solve_linear_constant(A, b, t0, t1, y0)
            if np.all(np.isfinite(y_end)):
                return y_end
        except Exception:
            pass
    
    # Fallback: Use a more robust method
    # Try implicit methods for stiff problems
    from scipy.integrate import solve_ivp
    
    try:
        sol = solve_ivp(f, [t0, t1], y0, method='Radau', rtol=1e-12, atol=1e-14, max_step=0.01)
        if sol.success and len(sol.t) > 0:
            return sol.y[:, -1]
    except Exception:
        pass
    
    try:
        sol = solve_ivp(f, [t0, t1], y0, method='BDF', rtol=1e-12, atol=1e-14, max_step=0.01)
        if sol.success and len(sol.t) > 0:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Last resort: RK4 with many steps
    return _solve_linear_nonconstant(f, t0, t1, y0, max_steps=20000)
# EVOLVE-BLOCK-END
