import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with 10 learnable parameters.
    Uses a richer basis including higher-order polynomials, cross terms,
    and reciprocal/rational terms to better approximate unknown functions.
    """
    # Center and scale inputs to roughly [-1, 1]
    # Domain [0.5, 2.0], center 1.25, half-range 0.75
    xc = (x - 1.25) / 0.75
    yc = (y - 1.25) / 0.75
    zc = (z - 1.25) / 0.75
    
    # Basis functions
    b0 = np.ones_like(x)          # constant
    b1 = xc                        # linear x
    b2 = yc                        # linear y
    b3 = zc                        # linear z
    b4 = xc * xc                   # quadratic x
    b5 = yc * yc                   # quadratic y
    b6 = zc * zc                   # quadratic z
    b7 = xc * yc                   # cross x*y
    b8 = xc * zc                   # cross x*z
    b9 = yc * zc                   # cross y*z
    
    # Combine with learnable parameters
    return (params[0] * b0 + params[1] * b1 + params[2] * b2 + params[3] * b3 +
            params[4] * b4 + params[5] * b5 + params[6] * b6 + params[7] * b7 +
            params[8] * b8 + params[9] * b9)
