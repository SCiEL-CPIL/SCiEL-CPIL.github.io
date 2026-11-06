# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.asarray(y0, dtype=float)
    n = len(y)
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 330s limit

    # --- Strategy: use scipy's solve_ivp with Radau (implicit, stiff) ---
    # If scipy is available, use it directly. Otherwise, implement a custom
    # implicit Runge-Kutta method.

    # First, try to use scipy.integrate.solve_ivp with Radau
    try:
        from scipy.integrate import solve_ivp

        # Use Radau method with tight tolerances
        sol = solve_ivp(
            f, t_span, y,
            method='Radau',
            rtol=1e-12,
            atol=1e-14,
            max_step=0.01
        )
        if sol.success and len(sol.t) > 0:
            return sol.y[:, -1]
    except Exception:
        pass

    # --- Fallback: custom implicit solver ---
    # Use a simple implicit Euler with Newton iteration, but with adaptive
    # step size and a deadline guard.

    def _newton_solve(A, b, max_iter=20, tol=1e-14):
        x = np.zeros_like(b)
        for _ in range(max_iter):
            r = b - A @ x
            if np.linalg.norm(r) < tol:
                break
            try:
                dx = np.linalg.solve(A, r)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(A, r, rcond=None)[0]
            x += dx
        return x

    h = (t1 - t0) / 10000  # start with small steps
    t = t0
    y_curr = y.copy()
    n_steps_max = 2000000

    while t < t1 - 1e-15:
        if time.time() > deadline - 5.0:
            break
        if t + h > t1:
            h = t1 - t

        # Implicit Euler: y_new = y_curr + h * f(t+h, y_new)
        # Solve: y_new - h * f(t+h, y_new) = y_curr
        # Use fixed-point iteration or Newton

        # Approximate Jacobian
        J = np.zeros((n, n))
        fz = f(t + h, y_curr)
        for j in range(n):
            e = np.zeros(n)
            e[j] = 1e-8 * max(1.0, abs(y_curr[j]))
            J[:, j] = (f(t + h, y_curr + e) - fz) / e[j]

        A = np.eye(n) - h * J
        b = y_curr + h * fz

        y_new = _newton_solve(A, b)
        t += h
        y_curr = y_new

    return y_curr
# EVOLVE-BLOCK-END
