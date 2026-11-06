"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Solve y' = f(t, y) on [t0, t1] for a 15-dim stiff linear system.
    Strategy: detect if f is linear and time-independent (y' = A@y + b),
    then compute y(t1) = e^{A*(t1-t0)} @ y0 + integral of e^{A*(t1-s)} b ds.
    Use matrix exponential via scipy if available, otherwise use a high-order
    Taylor/Padé approximation.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # 280s out of 300s limit

    # Try to detect linearity: sample f at a few points
    # If f(t, y) = A@y + b for constant A, b, we can exploit that.
    # Sample at t0 with y0, and at t0 with a perturbed y to get A columns.
    
    # First, get f(t0, y0)
    fy0 = f(t0, y0)
    
    # Check if f is time-independent and linear by sampling
    # Evaluate f at t1 with y0 to check time dependence
    fy0_t1 = f(t1, y0)
    if not np.allclose(fy0, fy0_t1, atol=1e-10):
        # Time-dependent; use a stiff ODE solver
        return _stiff_solve(f, t0, t1, y0, deadline)
    
    # Assume time-independent. Now extract A and b.
    # f(y) = A@y + b
    # Use finite differences to extract A
    # f(y + h*e_j) - f(y) ≈ h * A[:, j]
    
    eps = 1e-7
    A = np.zeros((n, n))
    b = fy0 - np.zeros(n)  # We'll compute b = f(y) - A@y once A is known
    
    for j in range(n):
        y_pert = y0.copy()
        y_pert[j] += eps
        fy_pert = f(t0, y_pert)
        A[:, j] = (fy_pert - fy0) / eps
    
    # Now b = fy0 - A @ y0
    b = fy0 - A @ y0
    
    # Verify linearity: check f(t0, 2*y0) ≈ 2*A@y0 + b
    y_check = 2.0 * y0
    fy_check = f(t0, y_check)
    predicted = A @ y_check + b
    if not np.allclose(fy_check, predicted, atol=1e-5):
        # Not linear; fall back to stiff solver
        return _stiff_solve(f, t0, t1, y0, deadline)
    
    # Now we have y' = A@y + b, constant.
    # Solution: y(t) = e^{A*(t-t0)} @ y0 + ∫_{t0}^{t} e^{A*(t-s)} b ds
    # = e^{A*dt} @ y0 + (e^{A*dt} - I) @ A^{-1} @ b  (if A invertible)
    # More robust: use the matrix exponential directly.
    
    # Compute e^{A*dt}
    try:
        from scipy.linalg import expm
        expA = expm(A * dt)
        # For the particular solution with constant b:
        # ∫_0^dt e^{A*s} b ds = solve(A, (e^{A*dt} - I) @ b) if A invertible
        # Or use the augmented matrix method:
        # [A I; 0 0] exponential gives the integral
        # Simpler: if A is invertible, particular = A^{-1} @ (e^{A*dt} - I) @ b
        try:
            # Check if A is singular
            detA = np.linalg.det(A)
            if abs(detA) > 1e-12:
                # Use A^{-1}
                A_inv = np.linalg.inv(A)
                particular = A_inv @ (expA - np.eye(n)) @ b
            else:
                # Use augmented matrix
                # [A I; 0 0] * dt
                # The top-right block of expm of this gives the integral
                aug = np.zeros((2*n, 2*n))
                aug[:n, :n] = A * dt
                aug[:n, n:] = np.eye(n) * dt
                exp_aug = expm(aug)
                particular = exp_aug[:n, n:] @ b
        except np.linalg.LinAlgError:
            # Use augmented matrix method
            aug = np.zeros((2*n, 2*n))
            aug[:n, :n] = A * dt
            aug[:n, n:] = np.eye(n) * dt
            exp_aug = expm(aug)
            particular = exp_aug[:n, n:] @ b
        
        y_final = expA @ y0 + particular
        return y_final
        
    except Exception:
        # Fallback to stiff solver
        return _stiff_solve(f, t0, t1, y0, deadline)


def _stiff_solve(f, t0, t1, y0, deadline):
    """Fallback: use scipy's solve_ivp with BDF method for stiff problems."""
    from scipy.integrate import solve_ivp
    
    y0 = np.array(y0, dtype=float)
    
    # Use a stiff solver with tight tolerances
    sol = solve_ivp(
        f, (t0, t1), y0,
        method='BDF',
        rtol=1e-10,
        atol=1e-12,
        max_step=(t1 - t0) / 100,
        t_eval=[t1]
    )
    
    if sol.success and len(sol.y) > 0:
        return sol.y[:, -1]
    
    # If BDF fails, try Radau
    sol = solve_ivp(
        f, (t0, t1), y0,
        method='Radau',
        rtol=1e-10,
        atol=1e-12,
        max_step=(t1 - t0) / 100,
        t_eval=[t1]
    )
    
    if sol.success and len(sol.y) > 0:
        return sol.y[:, -1]
    
    # Last resort: explicit RK4 with many steps
    return _rk4_fallback(f, t0, t1, y0, deadline)


def _rk4_fallback(f, t0, t1, y0, deadline):
    """Explicit RK4 as last resort."""
    y = np.array(y0, dtype=float)
    n_steps = 10000
    h = (t1 - t0) / n_steps
    t = t0
    for i in range(n_steps):
        if time.time() > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y
# EVOLVE-BLOCK-END
