"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Tensor-product Fourier basis with 10 terms.
    Basis:
      p0 (constant)
      p1*cos(2*pi*x)
      p2*sin(2*pi*x)
      p3*cos(2*pi*y)
      p4*sin(2*pi*y)
      p5*cos(4*pi*x)
      p6*sin(4*pi*x)
      p7*cos(4*pi*y)
      p8*sin(4*pi*y)
      p9*cos(2*pi*x)*cos(2*pi*y)
    
    This orthogonal basis captures periodic structure and smooth interactions.
    Linear in parameters, so BFGS converges to global minimum.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has at least 10 elements; pad with zeros if needed
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)))
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = p[:10]
    
    # Precompute trigonometric functions
    cos_2pi_x = np.cos(2 * np.pi * x)
    sin_2pi_x = np.sin(2 * np.pi * x)
    cos_2pi_y = np.cos(2 * np.pi * y)
    sin_2pi_y = np.sin(2 * np.pi * y)
    cos_4pi_x = np.cos(4 * np.pi * x)
    sin_4pi_x = np.sin(4 * np.pi * x)
    cos_4pi_y = np.cos(4 * np.pi * y)
    sin_4pi_y = np.sin(4 * np.pi * y)
    
    pred = (p0
            + p1 * cos_2pi_x
            + p2 * sin_2pi_x
            + p3 * cos_2pi_y
            + p4 * sin_2pi_y
            + p5 * cos_4pi_x
            + p6 * sin_4pi_x
            + p7 * cos_4pi_y
            + p8 * sin_4pi_y
            + p9 * cos_2pi_x * cos_2pi_y)
    
    return pred
# EVOLVE-BLOCK-END
