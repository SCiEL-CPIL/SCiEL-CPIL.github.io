"""Symbolic regression: nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from functools import lru_cache
import warnings

warnings.filterwarnings('ignore')


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for the acceleration of a damped nonlinear oscillator.
    Uses up to 10 fitted constants.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Linear restoring force
    linear_restoring = p0 * x

    # Cubic restoring force (Duffing)
    cubic_restoring = p1 * x**3

    # Linear damping
    linear_damping = p2 * v

    # Van der Pol damping term (x^2 * v)
    vdp_damping = p3 * x**2 * v

    # Cubic damping
    cubic_damping = p4 * v**3

    # Cross term x*v
    cross_xv = p5 * x * v

    # Time dependent forcing (cos)
    time_cos = p6 * np.cos(t)

    # Time dependent forcing (sin)
    time_sin = p7 * np.sin(t)

    # Parametric forcing x*cos(t)
    param_cos = p8 * x * np.cos(t)

    # Parametric forcing x*sin(t)
    param_sin = p9 * x * np.sin(t)

    return (linear_restoring + cubic_restoring + linear_damping + vdp_damping +
            cubic_damping + cross_xv + time_cos + time_sin + param_cos + param_sin)
# EVOLVE-BLOCK-END
