"""ODE solver: Heat2D 10x10 — combined spectral + adaptive approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = y0.size

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 285  # safety margin

    # Try the spectral/matrix-exponential approach first for linear problems.
    # This uses ~200 f-calls (Jacobian estimation) and is extremely accurate.
    try:
        result = _try_linear_expm(f, t0, t1, y0, start_time, deadline)
        if result is not None:
            return result
    except Exception:
        pass

    # Fallback: use scipy's Radau (implicit, L-stable) for stiff problems
    try:
        from scipy.integrate import solve_ivp

        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-10,
            atol=1e-12,
            max_step=(t1 - t0) / 50,
        )

        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Final fallback: manual RK4
    return _rk4_fallback(f, t0, t1, y0)


def _try_linear_expm(f, t0, t1, y0, start_time, deadline):
    """
    Attempt to solve y' = A*y + b using matrix exponential.
    Returns the final state if the problem is detected as linear/affine,
    otherwise returns None.
    """
    n = y0.size
    T = t1 - t0

    # Estimate Jacobian A via central differences (2n calls)
    # Use a moderate epsilon for numerical stability
    eps = 1e-6
    J = np.empty((n, n), dtype=float)
    fy0 = f(t0, y0)

    for j in range(n):
        e = eps * max(1.0, abs(y0[j]))
        yp = y0.copy()
        yp[j] += e
        ym = y0.copy()
        ym[j] -= e
        J[:, j] = (f(t0, yp) - f(t0, ym)) / (2 * e)

    # Check if we're still within deadline
    if time.time() > deadline:
        return None

    # Check if the problem is linear: f(t, y) ≈ A @ y
    # Test at a few points
    test_points = [
        np.zeros_like(y0),
        y0 * 0.5,
        y0 * 1.5,
        y0 + 0.1 * np.ones(n),
    ]

    max_resid = 0.0
    scale = max(1.0, np.linalg.norm(fy0))

    for yp in test_points:
        f_yp = f(t0, yp)
        pred = J @ yp
        resid = np.linalg.norm(f_yp - pred) / scale
        max_resid = max(max_resid, resid)

        if time.time() > deadline:
            return None

    # If residual is small, the problem is approximately linear
    if max_resid < 1e-5:
        # Solve y(T) = expm(A*T) @ y0
        try:
            from scipy.linalg import expm
            eA = expm(J * T)
            y_end = eA @ y0
            if np.all(np.isfinite(y_end)):
                return y_end
        except Exception:
            pass

    return None


def _rk4_fallback(f, t0, t1, y0):
    """Fallback RK4 implementation."""
    n_steps = 5000
    h = (t1 - t0) / n_steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            break
    return y
# EVOLVE-BLOCK-END
