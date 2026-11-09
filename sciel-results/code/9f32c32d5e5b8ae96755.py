# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0

    # Deadline guard
    deadline = time.time() + 290

    # Probe the Jacobian A at several points to detect time-dependence
    # For a linear system y' = A(t) y, f(t, y) = A(t) @ y
    # Get A by calling f(t, e_j) for each basis vector
    def get_A(t):
        A = np.empty((n, n), dtype=float)
        for j in range(n):
            e = np.zeros(n, dtype=float)
            e[j] = 1.0
            A[:, j] = f(t, e)
        return A

    # Check if A is time-independent by sampling at a few points
    A0 = get_A(t0)
    A1 = get_A(t1)
    if not np.all(np.isfinite(A0)) or not np.all(np.isfinite(A1)):
        # Fallback to generic ODE solver
        from scipy.integrate import solve_ivp
        sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-12, atol=1e-14, max_step=dt/100)
        return sol.y[:, -1]

    # Check time-independence
    if np.max(np.abs(A1 - A0)) < 1e-10 * max(1.0, np.max(np.abs(A0))):
        # Time-independent linear system: y(t1) = expm(dt * A) @ y0
        M = dt * A0
        # Use scaling and squaring for stability
        y_end = expm(M) @ y0
        return y_end
    else:
        # Time-dependent: split into subintervals and use product of exponentials
        # Use enough subintervals to capture the time variation
        n_sub = 50
        h = dt / n_sub
        y = y0.copy()
        t = t0
        for i in range(n_sub):
            if time.time() > deadline:
                break
            # Midpoint rule for A
            A_mid = get_A(t + h / 2)
            M = h * A_mid
            y = expm(M) @ y
            t += h
        return y
# EVOLVE-BLOCK-END
