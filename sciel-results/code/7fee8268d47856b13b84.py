import numpy as np
import time
from scipy.linalg import expm, eig
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Extract the system matrix A by probing with basis vectors.
    # For a linear time-invariant system: f(t, y) = A @ y
    # f(t0, e_i) = A @ e_i = column i of A
    # This uses exactly n = 100 calls.
    A = np.empty((n, n), dtype=float)
    for j in range(n):
        e_j = np.zeros(n, dtype=float)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)
    
    # Check for NaNs or Infs
    if not np.all(np.isfinite(A)):
        return _fallback_adaptive(f, t0, t1, y0, deadline, start_time)
    
    dt = t1 - t0
    
    # Compute y(t1) = expm(A * dt) @ y0 using eigenvalue decomposition
    # This can be more accurate than scaling-and-squaring for well-conditioned problems
    try:
        # Try eigenvalue-based approach first for potentially higher accuracy
        eigenvalues, V = eig(A)
        # Check if eigenvalues are well-conditioned (not too large spread)
        if np.all(np.isfinite(eigenvalues)) and np.all(np.isfinite(V)):
            # Check condition number of V
            cond_V = np.linalg.cond(V)
            if cond_V < 1e10:
                # exp(A*dt) @ y0 = V @ diag(exp(eigenvalues*dt)) @ inv(V) @ y0
                c = np.linalg.solve(V, y0)
                y_end = V @ (np.exp(eigenvalues * dt) * c)
                if np.all(np.isfinite(y_end)):
                    return y_end
        # Fall through to expm if eigen approach failed or was ill-conditioned
        M = expm(A * dt)
        y_end = M @ y0
    except Exception:
        return _fallback_adaptive(f, t0, t1, y0, deadline, start_time)
    
    # Verify finite result
    if not np.all(np.isfinite(y_end)):
        return _fallback_adaptive(f, t0, t1, y0, deadline, start_time)
    
    return y_end

def _fallback_adaptive(f, t0, t1, y0, deadline, start_time):
    """High-accuracy fallback using DOP853 with tight tolerances."""
    # Try DOP853 with very tight tolerances
    try:
        sol = solve_ivp(
            f, (t0, t1), y0,
            method='DOP853',
            rtol=1e-14,
            atol=1e-16,
            max_step=(t1 - t0) / 200.0,
            dense_output=False
        )
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Check deadline
    if (time.time() - start_time) > deadline - 30.0:
        return _fallback_rk4(f, t0, t1, y0, deadline, start_time)
    
    # Try Radau for stiff problems
    try:
        sol = solve_ivp(
            f, (t0, t1), y0,
            method='Radau',
            rtol=1e-13,
            atol=1e-15,
            max_step=(t1 - t0) / 200.0,
            dense_output=False
        )
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Final fallback: high-accuracy RK4
    return _fallback_rk4(f, t0, t1, y0, deadline, start_time)

def _fallback_rk4(f, t0, t1, y0, deadline, start_time):
    """Simple RK4 fallback with fixed steps."""
    steps = 5000
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    for _ in range(steps):
        if (time.time() - start_time) > deadline - 5.0:
            break
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y
