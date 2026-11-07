"""Structural redesign: rich basis with rational, polynomial, and interaction terms."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Basis: 1, x, y, z, 1/x, 1/y, 1/z, xy, xz, yz
    This captures:
      - Linear trends
      - Reciprocal/hyperbolic behavior (important for rational functions)
      - Pairwise interactions
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Avoid division by zero (inputs are in [0.5, 2] so safe, but guard anyway)
    x_safe = np.where(np.abs(x) < 1e-12, 1e-12, x)
    y_safe = np.where(np.abs(y) < 1e-12, 1e-12, y)
    z_safe = np.where(np.abs(z) < 1e-12, 1e-12, z)

    inv_x = 1.0 / x_safe
    inv_y = 1.0 / y_safe
    inv_z = 1.0 / z_safe

    return (p0
            + p1 * x
            + p2 * y
            + p3 * z
            + p4 * inv_x
            + p5 * inv_y
            + p6 * inv_z
            + p7 * x * y
            + p8 * x * z
            + p9 * y * z)
# EVOLVE-BLOCK-END
