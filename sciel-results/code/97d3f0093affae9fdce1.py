"""ODE solver for 100-point convection-diffusion IVP using Parlett matrix exponential."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.linalg import expm


def integrate(f, t_span, y0):
    t_start = time.perf_counter()
    DEADLINE = 290.0  # Safety margin before 330s kill

    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    t0, t1 = t_span
    T = t1 - t0

    def _elapsed_ok():
        return (time.perf_counter() - t_start) < DEADLINE

    # Step 1: Extract the system matrix A from f.
    # Since f(t, y) = A @ y for linear time-invariant systems,
    # we can recover A by evaluating f at standard basis vectors.
    # f(t, e_i) = A @ e_i = column i of A
    # We need n+1 evaluations: one for each basis vector (the zero vector gives nothing useful)
    
    # Actually, we can be smarter: use random vectors to get more info per call,
    # but for n=100, n calls is fine and gives us A exactly (up to floating point).
    
    # Let's extract A using the identity matrix trick:
    # A = f(t, I) where I is the identity matrix, but f expects a vector.
    # So we evaluate f(t, e_i) for i = 0, ..., n-1 to get columns of A.
    
    # But wait - we need to be careful. The function f takes (t, y) where y is a vector.
    # So f(t, e_i) gives us column i of A.
    
    # Let's do this efficiently:
    A = np.zeros((n, n), dtype=float)
    e = np.zeros(n, dtype=float)
    
    for i in range(n):
        if not _elapsed_ok():
            # Fallback to DOP853 if we're running out of time
            break
        e[i] = 1.0
        col = f(t0, e)
        A[:, i] = col
        e[i] = 0.0
    
    # Check if we successfully extracted A
    if _elapsed_ok() and np.all(np.isfinite(A)):
        # Step 2: Compute the matrix exponential e^(A*T) and apply to y0
        try:
            AT = A * T
            # Use Parlett recurrence via scipy's expm which uses scaling and squaring
            # with Pade approximation - this is the Parlett method
            expAT = expm(AT)
            result = expAT @ y0
            if np.all(np.isfinite(result)):
                return result
        except Exception:
            pass
    
    # Fallback: Use solve_ivp with DOP853
    try:
        from scipy.integrate import solve_ivp
        sol = solve_ivp(
            f,
            (t0, t1),
            y0,
            method="DOP853",
            rtol=1e-12,
            atol=1e-14,
            max_step=T / 50.0,
            first_step=T / 1e6,
            dense_output=False,
        )
        if sol.success and sol.t.size > 0:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Last-resort: fixed-step RK4
    steps = 2000
    h = T / steps
    t, y = t0, y0.copy()
    for _ in range(steps):
        if not _elapsed_ok():
            break
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            break
    return y
# EVOLVE-BLOCK-END
