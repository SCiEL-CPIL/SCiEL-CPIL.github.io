"""ODE solver: Heat2D 10x10 - integrate(f, t_span, y0)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _is_linear_autonomous(f, t0, y0):
    """
    Check if f(t, y) is linear and autonomous: f(t, y) = A @ y.
    We verify:
      1. f(t0, y0) == f(t0 + delta, y0)  (autonomy)
      2. f(t0, y0) == A @ y0 for some A, where A is determined by probing
         with random vectors.
    Returns A if linear autonomous, else None.
    """
    n = len(y0)
    
    # Check autonomy: f should not depend on t
    fy0 = f(t0, y0)
    fy0_t2 = f(t0 + 0.001, y0)
    if not np.allclose(fy0, fy0_t2, rtol=1e-12, atol=1e-14):
        return None
    
    # Probe with random vectors to determine A
    # f(t, v) should equal A @ v for any v
    rng = np.random.default_rng(42)
    
    # Use a set of probe vectors
    # First, try the zero vector: f(t, 0) should be 0 for linear
    zero_vec = np.zeros(n)
    fz = f(t0, zero_vec)
    if not np.allclose(fz, 0.0, atol=1e-14):
        return None
    
    # Determine A by probing with basis vectors and random vectors
    # We need n independent vectors to determine A
    probes = []
    results = []
    
    # Use y0 as first probe (if non-zero)
    if np.linalg.norm(y0) > 1e-15:
        probes.append(y0)
        results.append(fy0)
    
    # Add basis vectors
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        probes.append(e)
        results.append(f(t0, e))
    
    # We have more than n probes, use least squares to find A
    P = np.array(probes)  # shape (m, n)
    R = np.array(results)  # shape (m, n)
    
    # We want A such that A @ P.T = R.T, i.e., A @ p_i = r_i
    # So A = R.T @ P.T^{-1} if P is square, or use lstsq
    # A @ P.T = R.T  =>  A = R.T @ pinv(P.T)
    A = np.linalg.lstsq(P.T, R.T, rcond=None)[0].T
    
    # Verify: check that A @ v ≈ f(t0, v) for random test vectors
    for trial in range(5):
        v = rng.standard_normal(n)
        fv = f(t0, v)
        Av = A @ v
        if not np.allclose(fv, Av, rtol=1e-10, atol=1e-12):
            return None
    
    # Also verify with y0 itself
    Ay0 = A @ y0
    if not np.allclose(fy0, Ay0, rtol=1e-10, atol=1e-12):
        return None
    
    return A


def _solve_linear_heat(f, t0, t1, y0, A):
    """
    Solve y' = A @ y using matrix exponential.
    y(t) = exp((t-t0) * A) @ y0
    """
    from scipy.linalg import expm
    
    dt = t1 - t0
    M = expm(dt * A)
    return M @ y0


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    dt_total = t1 - t0
    deadline = time.time() + 280.0  # Safety margin before 330s kill

    # Try to detect if this is a linear autonomous system (heat equation discretization)
    A = _is_linear_autonomous(f, t0, y)
    if A is not None:
        result = _solve_linear_heat(f, t0, t1, y, A)
        if np.all(np.isfinite(result)):
            return result
    
    # Fallback: fixed-step RK45 (Dormand-Prince)
    # RK45 (Dormand-Prince) coefficients
    c2, c3, c4, c5, c6, c7 = 1/5, 3/10, 4/5, 8/9, 1, 1
    a21 = 1/5
    a31, a32 = 3/40, 9/40
    a41, a42, a43 = 44/45, -56/15, 32/9
    a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
    a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
    a71, a72, a73, a74, a75, a76 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84

    # 5th order solution coefficients (local extrapolation)
    b1, b2, b3, b4, b5, b6, b7 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0

    # Fixed step count
    n_steps = 30000
    h = dt_total / n_steps
    t = t0

    for i in range(n_steps):
        if time.time() > deadline:
            break

        k1 = f(t, y)
        k2 = f(t + c2 * h, y + h * a21 * k1)
        k3 = f(t + c3 * h, y + h * (a31 * k1 + a32 * k2))
        k4 = f(t + c4 * h, y + h * (a41 * k1 + a42 * k2 + a43 * k3))
        k5 = f(t + c5 * h, y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
        k6 = f(t + c6 * h, y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5))
        k7 = f(t + c7 * h, y + h * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6))

        y = y + h * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6 + b7 * k7)
        t += h

    return y
# EVOLVE-BLOCK-END
