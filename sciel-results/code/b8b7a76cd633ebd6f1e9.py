"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Acceleration of a damped nonlinear oscillator with driving force, from position and velocity."""
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    # Use a mixed nonlinear basis: cubic in x, quadratic in v, and cross terms.
    # This captures common oscillator physics: restoring force (x, x^3), damping (v, v^2, v^3),
    # and coupling (x*v, x^2*v, x*v^2).
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    xv = x * v
    x2v = x2 * v
    xv2 = x * v2
    # Reorder parameters to map to specific basis functions for a structured fit
    # p0: constant (bias)
    # p1: x (linear restoring)
    # p2: v (linear damping)
    # p3: x^3 (cubic stiffness)
    # p4: x*v (cross coupling)
    # p5: v^2 (quadratic damping)
    # p6: x^2*v (second-order cross)
    # p7: x*v^2 (second-order cross)
    # p8: x^2 (quadratic restoring - if system is asymmetric)
    # p9: v^3 (cubic damping)
    return (p0 + p1 * x + p2 * v + p3 * x3 + p4 * xv + p5 * v2 + p6 * x2v + p7 * xv2 + p8 * x2 + p9 * v3)
# EVOLVE-BLOCK-END
