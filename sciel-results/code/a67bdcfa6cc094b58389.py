import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator modeled as a complete cubic polynomial
    in scaled position and velocity.
    
    The basis functions are:
    1, X, V, X^2, V^2, XV, X^3, X^2V, XV^2, V^3
    where X and V are scaled inputs.
    """
    # Scale inputs to improve numerical conditioning.
    # Training ranges: x in [-0.57, 0.53], v in [-0.18, 0.28]
    # Scaling factors chosen to map inputs to roughly [-1, 1]
    X = x / 0.6
    V = v / 0.3
    
    # Precompute powers and products
    X2 = X * X
    X3 = X2 * X
    V2 = V * V
    V3 = V2 * V
    XV = X * V
    X2V = X2 * V
    XV2 = XV * V
    
    # Construct the linear combination of basis functions
    # params[0] * 1 + params[1] * X + params[2] * V + 
    # params[3] * X^2 + params[4] * V^2 + params[5] * XV +
    # params[6] * X^3 + params[7] * X^2V + params[8] * XV^2 +
    # params[9] * V^3
    
    return (params[0] + 
            params[1] * X + 
            params[2] * V + 
            params[3] * X2 + 
            params[4] * V2 + 
            params[5] * XV + 
            params[6] * X3 + 
            params[7] * X2V + 
            params[8] * XV2 + 
            params[9] * V3)
