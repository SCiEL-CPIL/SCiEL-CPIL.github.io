"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rich basis for symbolic regression with 3 inputs.
    Uses normalized inputs and a combination of:
      - Linear terms
      - Quadratic terms (pure and cross)
      - Cubic terms (pure)
      - Some interaction terms
      - Radial basis functions (Gaussian-like bumps)
      - Trigonometric terms (for periodic behavior)
    
    params has length 10. We use all 10 parameters.
    """
    p = params
    
    # Normalize inputs to [-1, 1] for better conditioning
    # Inputs are in [0.5, 2.0], so center=1.25, half_range=0.75
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75
    
    # Build basis functions
    # We have 10 parameters, so we need 10 basis terms
    
    # Terms 1-3: linear
    b1 = u
    b2 = v
    b3 = w
    
    # Terms 4-6: quadratic pure
    b4 = u * u
    b5 = v * v
    b6 = w * w
    
    # Terms 7-9: cross quadratic
    b7 = u * v
    b8 = v * w
    b9 = u * w
    
    # Term 10: a combination of higher-order or special function
    # Use a product of all three (cubic interaction)
    b10 = u * v * w
    
    return (p[0] * b1 + p[1] * b2 + p[2] * b3 +
            p[3] * b4 + p[4] * b5 + p[5] * b6 +
            p[6] * b7 + p[7] * b8 + p[8] * b9 +
            p[9] * b10)
# EVOLVE-BLOCK-END
