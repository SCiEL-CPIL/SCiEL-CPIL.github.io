"""ODE solver: 15-dimensional stiff linear system."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate a 15-dimensional stiff linear ODE y' = f(t, y) over t_span.
    Strategy:
      1. Detect if the system is linear with a constant coefficient matrix A.
         If so, compute the matrix exponential expm((t1-t0)*A) and apply it to y0.
         This is exact (up to floating point) for linear constant-coefficient systems.
      2. If detection fails or the system is nonlinear, fall back to Radau (implicit
         Runge-Kutta method suitable for stiff problems) with tight tolerances.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # Leave 50s margin before 330s kill

    # Try to detect linear constant-coefficient system: f(t, y) = A @ y
    # Sample at a few points to check linearity and time-independence
    try:
        # Check if f(t, y) = A @ y for some constant A
        # Sample at t0 with y0
        f0 = np.asarray(f(t0, y0), dtype=float)
        
        # If y0 is zero, we can't detect A from f(t, 0) = 0 alone.
        # Use a probe vector.
        probe = np.zeros(n)
        probe[0] = 1.0
        f_probe = np.asarray(f(t0, probe), dtype=float)
        
        # Check if f(t, y) is linear: f(t, y) = A @ y
        # For a linear system, f(t, y) = A @ y for all t
        # Check time-independence: f(t0, y) should equal f(t_mid, y)
        t_mid = (t0 + t1) / 2.0
        f0_mid = np.asarray(f(t_mid, y0), dtype=float)
        
        if not np.allclose(f0, f0_mid, rtol=1e-10, atol=1e-12):
            # Not time-invariant, or nonlinear
            raise ValueError("Not time-invariant")
        
        # Build A from probe vectors
        A = np.zeros((n, n))
        for j in range(n):
            ej = np.zeros(n)
            ej[j] = 1.0
            A[:, j] = np.asarray(f(t0, ej), dtype=float)
        
        # Verify linearity: f(t, y) = A @ y for a random test vector
        test_y = np.random.randn(n)
        f_test = np.asarray(f(t0, test_y), dtype=float)
        A_test = A @ test_y
        if not np.allclose(f_test, A_test, rtol=1e-10, atol=1e-12):
            raise ValueError("Not linear")
        
        # Constant coefficient system: y(t1) = expm((t1-t0)*A) @ y0
        dt = t1 - t0
        M = expm(dt * A)
        y_end = M @ y0
        
        if time.time() > deadline:
            return y_end
        
        return y_end
    
    except Exception:
        pass
    
    # Fallback: use Radau with tight tolerances
    try:
        sol = solve_ivp(
            f, t_span, y0,
            method='Radau',
            rtol=1e-12,
            atol=1e-14,
            max_step=(t1 - t0) / 500
        )
        if time.time() > deadline:
            # Return best available
            if sol.y.size > 0:
                return sol.y[:, -1]
            return y0
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Last resort: implicit midpoint or backward Euler
    try:
        sol = solve_ivp(
            f, t_span, y0,
            method='BDF',
            rtol=1e-10,
            atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    return y0
# EVOLVE-BLOCK-END
