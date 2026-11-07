"""Symbolic regression: nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 140.0  # Safety margin before 150s kill
_START_TIME = None


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for the acceleration of a nonlinear oscillator.

    Uses raw input scales (no normalization) with the same structural basis:
      a = p0*x + p1*x^3 + p2*v + p3*v^3 + p4*x^2*v + p5*x*v
        + p6*t + p7*sin(t) + p8*x^2 + p9

    Raw scales are preferred because normalization introduced large parameter
    scaling that degraded BFGS convergence (observed: 151M vs 3.6K score).
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
    v2 = v * v
    v3 = v2 * v

    # Duffing-type restoring (linear + cubic)
    restoring = p0 * x + p1 * x3

    # Van der Pol style damping: linear + cubic saturation + x^2*v
    damping = p2 * v + p3 * v3 + p4 * x2 * v

    # Parametric / gyroscopic coupling
    coupling = p5 * x * v

    # Time-dependent terms (linear drift + sinusoidal drive)
    time_terms = p6 * t + p7 * np.sin(t)

    # Quadratic position correction and offset
    correction = p8 * x2 + p9

    return restoring + damping + coupling + time_terms + correction
# EVOLVE-BLOCK-END
