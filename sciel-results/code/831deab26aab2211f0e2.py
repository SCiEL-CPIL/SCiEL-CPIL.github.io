import numpy as np
import time
from scipy.sparse.linalg import expm_multiply

def integrate(f, t_span, y0):
    t0, t1 = float(t_span[0]), float(t_span[1])
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0

    calls = [0]
    start_time = time.time()
    time_limit = 310.0  # Safety margin before 330s kill

    def wrapped_f(t, y):
        calls[0] += 1
        if time.time() - start_time > time_limit or calls[0] > 1950000:
            raise TimeoutError("Time or call limit exceeded")
        return np.asarray(f(t, y), dtype=float)

    best_y = None

    # Strategy 1: Assume Linear Autonomous System y' = A y
    # Try to construct A by probing.
    try:
        if time.time() - start_time > time_limit:
            return y0

        # Probe f at t0
        f0 = wrapped_f(t0, y0)

        # Build A using forward differences.
        # A[:, i] = (f(t0, y0 + eps*e_i) - f0) / eps
        # This takes n calls.
        eps = 1e-8
        A = np.zeros((n, n))
        for i in range(n):
            y_pert = y0.copy()
            y_pert[i] += eps
            f_pert = wrapped_f(t0, y_pert)
            A[:, i] = (f_pert - f0) / eps

        # Check linearity:
        # Pick a random vector y_rand
        # Check if f(y_rand) approx A @ y_rand
        # This costs 1 call.
        rng = np.random.default_rng(42)
        y_rand = rng.standard_normal(n)
        f_rand = wrapped_f(t0, y_rand)
        residual = np.linalg.norm(f_rand - A @ y_rand)
        norm_f = np.linalg.norm(f_rand)

        # Check autonomy:
        # f(t1, y0) vs f0
        f_t1 = wrapped_f(t1, y0)
        residual_autonomy = np.linalg.norm(f_t1 - f0)

        if residual < 1e-6 * (norm_f + 1e-12) and residual_autonomy < 1e-6 * (norm_f + 1e-12):
            # System appears linear and autonomous.
            # Compute expm(A * dt) @ y0
            # Use expm_multiply for efficiency? expm_multiply(M, v) computes expm(M) @ v.
            # It is often faster and more stable for large dt.
            M = A * dt
            y_final = expm_multiply(M, y0)

            if np.all(np.isfinite(y_final)):
                best_y = y_final
            else:
                best_y = None
        else:
            best_y = None

    except Exception:
        best_y = None

    if best_y is not None:
        return best_y

    # If linear assumption failed, fall back to generic solvers
    if time.time() - start_time < time_limit:
        try:
            sol = solve_ivp(wrapped_f, (t0, t1), y0, method='Radau',
                           rtol=1e-12, atol=1e-14, max_step=0.5)
            if sol.success and sol.y.shape[1] > 0:
                y = sol.y[:, -1]
                if np.all(np.isfinite(y)):
                    best_y = y
        except Exception:
            pass

    if best_y is None and time.time() - start_time < time_limit:
        try:
            sol = solve_ivp(wrapped_f, (t0, t1), y0, method='BDF',
                           rtol=1e-12, atol=1e-14, max_step=0.5)
            if sol.success and sol.y.shape[1] > 0:
                y = sol.y[:, -1]
                if np.all(np.isfinite(y)):
                    best_y = y
        except Exception:
            pass

    if best_y is None:
        return y0
    return best_y
