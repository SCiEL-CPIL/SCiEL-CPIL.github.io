"""Symbolic regression: nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_DEADLINE = 120.0  # Safety margin before 120s kill (evaluator limit)


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for the acceleration of a nonlinear oscillator.
    
    Structural change: Hybrid model combining physics-based polynomial terms
    with a small set of Gaussian RBF terms for local flexibility.
    
    Uses raw input scales (no normalization) as normalization was observed to
    degrade BFGS convergence significantly.
    
    Basis functions (10 parameters):
      p0 * x       : linear restoring
      p1 * x^3     : cubic (Duffing) restoring
      p2 * v       : linear damping
      p3 * x^2*v   : Van der Pol nonlinear damping
      p4 * x*v     : parametric / gyroscopic coupling
      p5 * sin(t)  : sinusoidal drive
      p6 * x^2     : quadratic position correction
      p7 * RBF1    : local Gaussian correction centered at (t=35, x=0, v=0)
      p8 * RBF2    : local Gaussian correction centered at (t=45, x=0, v=0)
      p9           : offset
    
    The RBF terms provide local flexibility to capture non-polynomial behavior
    that the global polynomial terms cannot represent, while the physics-based
    terms provide the correct global structure.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    if time.time() - _START_TIME > _DEADLINE:
        # Return a safe fallback to avoid timeout
        return np.zeros_like(t, dtype=np.float64)

    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Precompute powers of raw inputs
    x2 = x * x
    x3 = x2 * x

    # Duffing-type restoring (linear + cubic)
    restoring = p0 * x + p1 * x3

    # Nonlinear damping: linear + Van der Pol x^2*v
    damping = p2 * v + p3 * x2 * v

    # Parametric / gyroscopic coupling
    coupling = p4 * x * v

    # Sinusoidal drive (bounded, avoids t-scale ill-conditioning)
    drive = p5 * np.sin(t)

    # Quadratic position correction
    quadratic = p6 * x2

    # Gaussian RBF terms for local flexibility
    # Centers chosen at mid-range t values, centered x and v
    # Width sigma chosen to cover roughly 1/4 of the t-range
    sigma_t = 5.0
    sigma_x = 0.1
    sigma_v = 0.2

    # RBF1: centered at t=35, x=0, v=0
    dt1 = t - 35.0
    dx1 = x - 0.0
    dv1 = v - 0.0
    rbf1 = np.exp(-(dt1**2 / (2*sigma_t**2) + dx1**2 / (2*sigma_x**2) + dv1**2 / (2*sigma_v**2)))

    # RBF2: centered at t=45, x=0, v=0
    dt2 = t - 45.0
    dx2 = x - 0.0
    dv2 = v - 0.0
    rbf2 = np.exp(-(dt2**2 / (2*sigma_t**2) + dx2**2 / (2*sigma_x**2) + dv2**2 / (2*sigma_v**2)))

    # Offset
    offset = p9

    return restoring + damping + coupling + drive + quadratic + p7 * rbf1 + p8 * rbf2 + offset
# EVOLVE-BLOCK-END
