"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational function approximation with even-powered denominator for stability:
    y = (p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4) / (1 + q1*x^2 + q2*x^4 + q3*x^6)
    
    params: [p0, p1, p2, p3, p4, q1, q2, q3, r1, r2]
    The denominator uses only even powers, ensuring it is bounded away from zero
    for reasonable parameter values, which aids BFGS convergence.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    q1 = params[5]
    q2 = params[6]
    q3 = params[7]
    
    # Compute powers of x
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x6 = x3 * x3
    
    # Compute numerator: p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4
    numerator = p0 + p1 * x + p2 * x2 + p3 * x3 + p4 * x4
    
    # Compute denominator: 1 + q1*x^2 + q2*x^4 + q3*x^6
    # This is always positive for |x| <= 2 if q1, q2, q3 are not too negative
    denominator = 1.0 + q1 * x2 + q2 * x4 + q3 * x6
    
    # Ensure denominator is bounded away from zero for numerical stability
    eps = 1e-8
    safe_denominator = np.where(np.abs(denominator) < eps, eps * np.sign(denominator + 1e-15), denominator)
    
    result = numerator / safe_denominator
    
    # Ensure result is finite
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
