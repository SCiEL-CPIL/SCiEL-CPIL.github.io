"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm


def integrate(f, t_span, y0):
    """
    Solve the fixed 10x10 2D heat equation IVP on [0, 0.05].
    Strategy: infer the linear operator A (and any time-scaling) from f,
    then compute the exact solution via matrix exponential.
    Falls back to RK4 if inference fails.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # safety margin before 300s

    # Try to infer the linear operator A such that f(t, y) = A @ y
    # The heat equation is linear: f(t, y) = c * L @ y where L is the 2D Laplacian.
    # We probe f at t=0 with a few vectors to reconstruct A.

    # Use n orthogonal probe vectors (standard basis) to get columns of A.
    # But that's n calls. Instead, use a random probe and verify linearity.
    # Actually, for a 100-dim system, using n=100 basis vectors costs 100 calls.
    # Alternative: use 2 random vectors to check linearity, then use structure.

    # Better approach: The 2D heat equation on a 10x10 grid has a known Laplacian structure.
    # But we don't know the exact boundary conditions or scaling. Let's infer A.

    # Use n basis vectors to get A exactly. This costs n calls.
    # Then compute expm(A * dt) @ y0. Total calls: n + 0 = 100.
    # Score: -log10(rel_err) - 0.5*log10(100). With machine precision, rel_err ~ 1e-15,
    # so -log10(1e-15) = 15, and -0.5*log10(100) = -1. Score ~ 14.

    # But we can do better: use fewer probes if we assume the structure.
    # Let's try: probe with y0 itself and a few other vectors to verify linearity.

    # First, check if f is linear: f(t, a*y + b*z) == a*f(t,y) + b*f(t,z)
    # Probe 1: f(0, y0)
    fy0 = f(t0, y0)

    # Probe 2: f(0, 2*y0)
    fy0_2 = f(t0, 2.0 * y0)

    # Check linearity: f(0, 2*y0) should be 2*f(0, y0)
    lin_check = np.linalg.norm(fy0_2 - 2.0 * fy0) / (np.linalg.norm(fy0) + 1e-30)
    if lin_check > 1e-10:
        # Not linear? Fall back to RK4
        return _rk4_safe(f, t0, t1, y0, deadline)

    # It's linear. Now infer A.
    # Use n basis vectors. n=100 for 10x10 grid.
    # Build A column by column.
    A = np.empty((n, n), dtype=float)
    for j in range(n):
        if time.time() > deadline:
            return _rk4_safe(f, t0, t1, y0, deadline)
        e_j = np.zeros(n, dtype=float)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)

    # Now A is the full linear operator. f(t, y) = A @ y (time-independent).
    # Compute exact solution: y(t1) = expm(A * dt) @ y0
    if time.time() > deadline:
        return _rk4_safe(f, t0, t1, y0, deadline)

    try:
        M = A * dt
        Y = expm(M)
        y_final = Y @ y0
        if np.all(np.isfinite(y_final)):
            return y_final
    except Exception:
        pass

    # Fallback
    return _rk4_safe(f, t0, t1, y0, deadline)


def _rk4_safe(f, t0, t1, y0, deadline):
    """RK4 with adaptive step and deadline guard."""
    y = np.array(y0, dtype=float)
    t = t0
    dt_total = t1 - t0

    # Start with a reasonable number of steps
    steps = 1000
    h = dt_total / steps

    while steps <= 100000:
        y_trial = np.array(y0, dtype=float)
        t_trial = t0
        success = True
        for _ in range(steps):
            if time.time() > deadline:
                return y_trial  # return best so far
            k1 = f(t_trial, y_trial)
            k2 = f(t_trial + h / 2, y_trial + h / 2 * k1)
            k3 = f(t_trial + h / 2, y_trial + h / 2 * k2)
            k4 = f(t_trial + h, y_trial + h * k3)
            y_trial = y_trial + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t_trial += h
            if not np.all(np.isfinite(y_trial)) or np.max(np.abs(y_trial)) > 1e12:
                success = False
                break
        if success:
            return y_trial
        # Increase steps for stability
        steps *= 2
        h = dt_total / steps

    return y
# EVOLVE-BLOCK-END
