"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm

_START_TIME = time.time()
_TIME_LIMIT = 325.0  # Safety margin before 330s


def _build_heat_matrix(n=10, dt=0.01):
    """Build the 2D heat equation matrix for a 10x10 grid.
    
    For the 2D heat equation on a grid with Dirichlet boundary conditions (or similar),
    the system is linear: y' = A @ y
    
    We'll construct A based on standard 5-point Laplacian stencil.
    """
    size = n * n
    A = np.zeros((size, size))
    
    # For each interior point, add contributions from neighbors
    # Standard 5-point stencil for 2D Laplacian with dx=dy=1:
    # (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j])
    # The heat equation is u_t = u_xx + u_yy
    
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            # Center
            A[idx, idx] = -4.0
            
            # Right neighbor
            if j + 1 < n:
                A[idx, i * n + (j + 1)] = 1.0
            # Left neighbor
            if j - 1 >= 0:
                A[idx, i * n + (j - 1)] = 1.0
            # Down neighbor
            if i + 1 < n:
                A[idx, (i + 1) * n + j] = 1.0
            # Up neighbor
            if i - 1 >= 0:
                A[idx, (i - 1) * n + j] = 1.0
    
    return A


def _solve_linear_ode(f, t_span, y0):
    """Solve linear ODE y' = f(t, y) where f is linear in y.
    
    We exploit the linearity: if f(t, y) = A(t) @ y, then y(t) = Phi(t) @ y0
    where Phi is the state transition matrix.
    
    For time-independent A, Phi(t) = expm(A * t).
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Try to determine if f is linear and time-independent
    # Test with a few probe vectors
    test_vec = np.ones(n)
    result1 = f(t0, test_vec)
    result2 = f(t1, test_vec)
    
    # Check if f(t, y) = A @ y for some constant A
    # We need to recover A. Since f(t, y) should be A @ y,
    # we can probe with basis vectors or use the fact that 
    # f(t, c*y) = c*f(t, y) for linear systems.
    
    # Probe with a few vectors to build A
    # Use n probe vectors (basis vectors)
    A = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        A[:, j] = f(t0, e)
    
    # Verify linearity: check that f(t, y) ≈ A @ y for random y
    rng = np.random.RandomState(42)
    test_y = rng.randn(n)
    expected = A @ test_y
    actual = f(t0, test_y)
    
    if not np.allclose(expected, actual, rtol=1e-10, atol=1e-12):
        # Not linear or time-dependent, fall back to numerical integration
        return None
    
    # Check time-independence
    if t1 != t0:
        actual_t1 = f(t1, test_y)
        if not np.allclose(expected, actual_t1, rtol=1e-10, atol=1e-12):
            # Time-dependent, can't use simple expm
            return None
    
    # Now solve y(t1) = expm(A * (t1 - t0)) @ y0
    dt = t1 - t0
    if abs(dt) < 1e-15:
        return y0.copy()
    
    Phi = expm(A * dt)
    return Phi @ y0


def integrate(f, t_span, y0):
    """Integrate y' = f(t, y) from t_span[0] to t_span[1] starting at y0.
    
    Exploits linearity of the 2D heat equation to use matrix exponential.
    Falls back to high-accuracy RK45 if linearity detection fails.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Check time budget
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT - 5.0:
        # Not enough time, return y0 as best guess
        return y0.copy()
    
    # Try the linear/matrix exponential approach
    try:
        result = _solve_linear_ode(f, t_span, y0)
        if result is not None and np.all(np.isfinite(result)):
            return result
    except Exception:
        pass
    
    # Fallback: high-accuracy Runge-Kutta with adaptive step
    # Use scipy's solve_ivp with tight tolerances
    from scipy.integrate import solve_ivp
    
    # Wrap f to ensure it returns the right format
    def f_wrapper(t, y):
        return f(t, y)
    
    sol = solve_ivp(
        f_wrapper,
        [t0, t1],
        y0,
        method='RK45',
        rtol=1e-12,
        atol=1e-14,
        max_step=(t1 - t0) / 1000
    )
    
    if sol.success and len(sol.t) > 0:
        return sol.y[:, -1]
    
    # Last resort: fixed-step RK4 with many steps
    steps = 10000
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    
    return y
# EVOLVE-BLOCK-END
