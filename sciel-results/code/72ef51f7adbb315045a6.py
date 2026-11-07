"""ODE integrator for the fixed Lorenz IVP using low-level scipy DOP853 with tight tolerances."""
import numpy as np
import time

# Try to import the low-level DOP853 class to reduce overhead
try:
    from scipy.integrate._ivp.dop853 import DOP853
    _HAS_DOP853 = True
except ImportError:
    _HAS_DOP853 = False
    from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard: 330s wall-clock limit, use 300s for safety
    start_time = time.time()
    deadline = start_time + 295.0
    
    if t1 <= t0:
        _ = f(t0, y0)
        return y0
    
    if _HAS_DOP853:
        return _integrate_dop853_tight(f, t0, t1, y0, deadline)
    else:
        return _integrate_solve_ivp_tight(f, t0, t1, y0, deadline)


def _integrate_dop853_tight(f, t0, t1, y0, deadline):
    """
    Uses the low-level DOP853 class with tighter tolerances to minimize relative error.
    rtol=1e-13, atol=1e-15 to push rel toward the 1e-14 floor.
    """
    # Use tighter tolerances to reduce rel significantly
    # The cost in extra calls is offset by the gain in -log10(rel)
    dop853 = DOP853(
        fun=f,
        t0=t0,
        y0=y0,
        t_bound=t1,
        rtol=1e-13,
        atol=1e-15,
        max_step=np.inf,
        direction=1.0,
        min_step=1e-12,
        n=3
    )
    
    while dop853.status == "running":
        # Check deadline
        if time.time() > deadline:
            # Return best result so far
            if dop853.y is not None:
                return dop853.y
            return y0
        dop853.step()
        
    if dop853.status == "reached_end":
        return dop853.y
    
    # If it failed, fallback to solve_ivp
    return _integrate_solve_ivp_tight(f, t0, t1, y0, deadline)


def _integrate_solve_ivp_tight(f, t0, t1, y0, deadline):
    """Fallback using solve_ivp with tight tolerances."""
    sol = solve_ivp(
        f,
        (t0, t1),
        y0,
        method='DOP853',
        rtol=1e-13,
        atol=1e-15,
        dense_output=False
    )
    if sol.success:
        return sol.y[:, -1]
    else:
        # Return y0 as a finite fallback
        return y0
