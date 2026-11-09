"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Richer basis with shifted Legendre polynomials up to degree 3 in each variable,
    plus key interaction and non-linear terms. This provides better conditioning
    and captures smoother non-polynomial trends.
    """
    # Shift inputs to [0, 1] for better-conditioned Legendre polynomials
    # Original domain: [0.5, 2], shift to [0, 1]
    xs = (x - 0.5) / 1.5  # maps [0.5, 2] -> [0, 1]
    ys = (y - 0.5) / 1.5
    zs = (z - 0.5) / 1.5

    # Shifted Legendre polynomials on [0, 1]:
    # L0(t) = 1
    # L1(t) = 2t - 1
    # L2(t) = 6t^2 - 6t + 1
    # L3(t) = 20t^3 - 30t^2 + 12t - 1

    # Compute Legendre polynomials for each variable
    L0x = np.ones_like(xs)
    L1x = 2 * xs - 1
    L2x = 6 * xs**2 - 6 * xs + 1
    L3x = 20 * xs**3 - 30 * xs**2 + 12 * xs - 1

    L0y = np.ones_like(ys)
    L1y = 2 * ys - 1
    L2y = 6 * ys**2 - 6 * ys + 1
    L3y = 20 * ys**3 - 30 * ys**2 + 12 * ys - 1

    L0z = np.ones_like(zs)
    L1z = 2 * zs - 1
    L2z = 6 * zs**2 - 6 * zs + 1
    L3z = 20 * zs**3 - 30 * zs**2 + 12 * zs - 1

    # Build 10-term basis:
    # p0: L0x * L0y * L0z = 1 (constant)
    # p1: L1x * L0y * L0z = 2xs - 1 (linear in x)
    # p2: L0x * L1y * L0z = 2ys - 1 (linear in y)
    # p3: L0x * L0y * L1z = 2zs - 1 (linear in z)
    # p4: L2x * L0y * L0z = 6xs^2 - 6xs + 1 (quadratic in x)
    # p5: L0x * L2y * L0z = 6ys^2 - 6ys + 1 (quadratic in y)
    # p6: L0x * L0y * L2z = 6zs^2 - 6zs + 1 (quadratic in z)
    # p7: L1x * L1y * L0z = (2xs-1)(2ys-1) (interaction x*y)
    # p8: L1x * L0y * L1z = (2xs-1)(2zs-1) (interaction x*z)
    # p9: L0x * L1y * L1z = (2ys-1)(2zs-1) (interaction y*z)

    pred = (params[0] * L0x * L0y * L0z +
            params[1] * L1x * L0y * L0z +
            params[2] * L0x * L1y * L0z +
            params[3] * L0x * L0y * L1z +
            params[4] * L2x * L0y * L0z +
            params[5] * L0x * L2y * L0z +
            params[6] * L0x * L0y * L2z +
            params[7] * L1x * L1y * L0z +
            params[8] * L1x * L0y * L1z +
            params[9] * L0x * L1y * L1z)

    return pred
# EVOLVE-BLOCK-END
