"""Symbolic regression: nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Self-sustained nonlinear oscillator (autonomous dynamics).
    
    Models a Van der Pol + Duffing type system with:
    - Duffing position nonlinearity: x + x^3
    - Van der Pol negative damping with cubic velocity saturation: v + v^3
    - Cross-coupling terms for richer limit-cycle behavior
    
    No explicit time-dependent driving terms; the system is autonomous.
    
    Normalized inputs for better BFGS conditioning:
      xn = x / 0.2
      vn = v / 0.35
    
    Basis functions (all autonomous, no t-dependence):
      0: 1                (Offset)
      1: xn               (Linear restoring force)
      2: xn^3             (Duffing cubic restoring)
      3: vn               (Linear damping / anti-damping)
      4: vn^3             (Cubic velocity saturation)
      5: xn * vn          (State-dependent damping)
      6: xn^2 * vn        (Quadratic position modulated damping)
      7: x^2              (Nonlinear position coupling)
      8: v^2              (Kinetic energy term)
      9: xn * vn^2        (Position-modulated quadratic velocity)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Normalization for conditioning
    xn = x / 0.2
    vn = v / 0.35

    # Precompute powers
    xn2 = xn * xn
    xn3 = xn2 * xn
    vn2 = vn * vn
    vn3 = vn2 * vn

    # Basis functions (autonomous - no explicit t)
    b0 = 1.0
    b1 = xn
    b2 = xn3
    b3 = vn
    b4 = vn3
    b5 = xn * vn
    b6 = xn2 * vn
    b7 = xn2
    b8 = vn2
    b9 = xn * vn2

    # Linear combination
    return (p0 * b0 +
            p1 * b1 +
            p2 * b2 +
            p3 * b3 +
            p4 * b4 +
            p5 * b5 +
            p6 * b6 +
            p7 * b7 +
            p8 * b8 +
            p9 * b9)
# EVOLVE-BLOCK-END
