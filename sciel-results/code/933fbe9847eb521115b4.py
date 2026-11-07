"""Symbolic regression: nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for the acceleration of a nonlinear oscillator.
    
    Design: Van der Pol + Duffing hybrid with autonomous dynamics only.
    No explicit time-dependent driving terms (sin(t), t-drift) to let
    parameters capture intrinsic oscillator behavior.
    
    Uses raw input scales (no normalization) based on evidence that
    normalization severely hurt BFGS convergence (score 151867834 vs
    ~3000-4400 for raw-scale variants).
    
    Basis functions (10 terms):
      p0 * x          - linear restoring
      p1 * x^3        - Duffing cubic restoring
      p2 * v          - linear damping
      p3 * v^3        - cubic velocity saturation (Van der Pol limit cycle)
      p4 * x^2 * v    - Van der Pol nonlinear damping
      p5 * x * v      - gyroscopic / parametric coupling
      p6 * x^2        - quadratic position correction
      p7 * v^2        - quadratic velocity correction
      p8 * x * v^2    - mixed cubic coupling
      p9               - offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Precompute powers using raw inputs
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v

    # Duffing-type restoring (linear + cubic)
    restoring = p0 * x + p1 * x3

    # Van der Pol style damping: linear + cubic saturation + x^2*v
    damping = p2 * v + p3 * v3 + p4 * x2 * v

    # Coupling terms
    coupling = p5 * x * v

    # Quadratic and mixed corrections
    corrections = p6 * x2 + p7 * v2 + p8 * x * v2

    # Offset
    offset = p9

    return restoring + damping + coupling + corrections + offset
# EVOLVE-BLOCK-END
