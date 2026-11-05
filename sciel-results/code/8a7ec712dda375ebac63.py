"""Rewritten program: ODE solver for 15-dim stiff linear system using matrix exponential."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm


def _extract_matrix_A(f, t0, n):
    """Extract the matrix A from f(t, y) = A @ y by probing with basis vectors."""
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e = np.zeros(n, dtype=float)
        e[j] = 1.0
        # Call f at (t0, e) to get column j of A
        col = np.asarray(f(t0, e), dtype=float)
        A[:, j] = col
    return A


def _verify_linearity(f, t0, A, n, num_checks=5):
    """Verify that f(t0, y) ≈ A @ y for random test vectors."""
    rng = np.random.default_rng(42)
    for _ in range(num_checks):
        y_test = rng.standard_normal(n)
        f_val = np.asarray(f(t0, y_test), dtype=float)
        pred = A @ y_test
        err = np.linalg.norm(f_val - pred)
        scale = max(1.0, np.linalg.norm(f_val))
        if err / scale > 1e-10:
            return False
    return True


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    # Extract matrix A
    A = _extract_matrix_A(f, t0, n)
    
    # Verify linearity at t0
    is_linear = _verify_linearity(f, t0, A, n)
    
    if is_linear:
        # Check if A is constant (doesn't depend on t) by sampling at another time
        A2 = _extract_matrix_A(f, t1, n)
        if np.linalg.norm(A - A2) / max(1.0, np.linalg.norm(A)) < 1e-10:
            # Constant coefficient linear system: use matrix exponential
            dt = t1 - t0
            try:
                M = expm(A * dt)
                y_final = M @ y0
                if np.all(np.isfinite(y_final)):
                    return y_final
            except Exception:
                pass
    
    # Fallback: use a robust implicit solver if matrix exponential fails or system isn't constant-linear
    # Use Radau or BDF from scipy for stiff problems
    from scipy.integrate import solve_ivp
    
    def f_wrapper(t, y):
        return np.asarray(f(t, y), dtype=float)
    
    try:
        sol = solve_ivp(
            f_wrapper, [t0, t1], y0,
            method='Radau',
            rtol=1e-12, atol=1e-14,
            max_step=(t1 - t0) / 100
        )
        if sol.success and sol.t[-1] == t1:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Last resort: simple backward Euler with many steps
    h = (t1 - t0) / 5000
    t = t0
    y = y0.copy()
    for _ in range(5000):
        if time.time() > deadline:
            break
        z = y.copy()
        for _newton in range(8):
            fz = np.asarray(f(t + h, z), dtype=float)
            J = np.empty((n, n))
            for j in range(n):
                e = np.zeros(n)
                e[j] = 1e-8 * max(1.0, abs(z[j]))
                J[:, j] = (np.asarray(f(t + h, z + e), dtype=float) - fz) / e[j]
            try:
                step = np.linalg.solve(np.eye(n) - h * J, -(z - y - h * fz))
            except np.linalg.LinAlgError:
                break
            z = z + step
            if np.linalg.norm(step) <= 1e-12 * (1 + np.linalg.norm(z)):
                break
        y = z
        t += h
    
    return y
# EVOLVE-BLOCK-END
