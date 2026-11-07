"""Symbolic regression: nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for the acceleration of a damped nonlinear oscillator
    with time-dependent driving.

    Structural change from seed (linear model):
    - Replaces linear-in-t with a bounded sinusoidal drive sin(t) to better
      represent periodic forcing without the ill-conditioning of large t values.
    - Adds Duffing cubic restoring (x^3) for nonlinear position dependence.
    - Adds Van der Pol style nonlinear damping (x^2 * v) and cubic velocity term (v^3).
    - Adds parametric coupling (x * v) for gyroscopic/parametric effects.
    - Retains linear terms in x, v, and a constant offset.
    - Uses raw input scales (no normalization) as normalization was observed to
      degrade BFGS convergence significantly in this lineage.

    Basis functions (10 parameters):
      p0 * x       : linear restoring
      p1 * x^3     : cubic (Duffing) restoring
      p2 * v       : linear damping
      p3 * v^3     : cubic velocity saturation
      p4 * x^2*v   : Van der Pol nonlinear damping
      p5 * x*v     : parametric / gyroscopic coupling
      p6 * sin(t)  : sinusoidal drive
      p7 * x^2     : quadratic position correction
      p8 * v^2     : quadratic velocity correction
      p9           : offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Precompute powers
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v

    # Duffing-type restoring (linear + cubic)
    restoring = p0 * x + p1 * x3

    # Nonlinear damping: linear + cubic saturation + Van der Pol x^2*v
    damping = p2 * v + p3 * v3 + p4 * x2 * v

    # Parametric / gyroscopic coupling
    coupling = p5 * x * v

    # Sinusoidal drive (bounded, avoids t-scale ill-conditioning)
    drive = p6 * np.sin(t)

    # Quadratic corrections
    quadratic = p7 * x2 + p8 * v2

    # Offset
    offset = p9

    return restoring + damping + coupling + drive + quadratic + offset
# EVOLVE-BLOCK-END
