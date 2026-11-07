"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 145.0  # safety margin before 150s kill


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Multiplicative parametric form: product of two affine linear terms
    in centered x, y, z, plus two linear correction terms. 10 parameters total.
    
    f = (p0*xc + p1*yc + p2*zc + p3) * (p4*xc + p5*yc + p6*zc + p7) + p8*xc + p9*yc
    
    This can exactly represent separable/bilinear structures that an additive
    quadratic polynomial approximates poorly.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = params

    # Center inputs for better numerical conditioning
    xc = x - 1.25
    yc = y - 1.25
    zc = z - 1.25

    t1 = p[0] * xc + p[1] * yc + p[2] * zc + p[3]
    t2 = p[4] * xc + p[5] * yc + p[6] * zc + p[7]

    return t1 * t2 + p[8] * xc + p[9] * yc
# EVOLVE-BLOCK-END
