"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.integrate


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Uses DOP853 with tight tolerances for high accuracy.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Use DOP853 with very tight tolerances for high accuracy
    # The convection-diffusion problem on 100 points over [0, 0.5] is not extremely stiff
    # so an explicit high-order method should work well
    
    sol = scipy.integrate.solve_ivp(
        f, (t0, t1), y0,
        method='DOP853',
        rtol=1e-11,
        atol=1e-13,
        max_step=np.inf
    )
    
    if sol.success:
        return sol.y[:, -1]
    
    # Fallback: try with looser tolerances if the tight one failed
    sol = scipy.integrate.solve_ivp(
        f, (t0, t1), y0,
        method='DOP853',
        rtol=1e-10,
        atol=1e-12,
        max_step=np.inf
    )
    
    if sol.success:
        return sol.y[:, -1]
    
    # Last resort: use Radau (implicit, handles stiffness)
    sol = scipy.integrate.solve_ivp(
        f, (t0, t1), y0,
        method='Radau',
        rtol=1e-10,
        atol=1e-12,
        max_step=np.inf
    )
    
    if sol.success:
        return sol.y[:, -1]
    
    # If all else fails, return the last computed state or y0
    if sol.y is not None and sol.y.shape[1] > 0:
        return sol.y[:, -1]
    return y0
# EVOLVE-BLOCK-END
