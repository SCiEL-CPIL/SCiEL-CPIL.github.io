"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational function model: y(x) = P_4(x) / Q_4(x)
    
    params[0:5]  -> coefficients of numerator P_4(x) = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4
    params[5:10] -> coefficients of denominator Q_4(x) = b0 + b1*x + b2*x^2 + b3*x^3 + b4*x^4
    
    We add a small regularization to the denominator to avoid division by zero.
    """
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Extract coefficients
    a0, a1, a2, a3, a4 = p[0], p[1], p[2], p[3], p[4]
    b0, b1, b2, b3, b4 = p[5], p[6], p[7], p[8], p[9]
    
    # Compute powers of x
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    # Numerator: P_4(x)
    numerator = a0 + a1 * x + a2 * x2 + a3 * x3 + a4 * x4
    
    # Denominator: Q_4(x)
    denominator = b0 + b1 * x + b2 * x2 + b3 * x3 + b4 * x4
    
    # Add small epsilon to avoid division by zero
    eps = 1e-12
    result = numerator / (denominator + eps)
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
