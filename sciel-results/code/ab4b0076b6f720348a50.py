"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: P5(x) * Q3(x)
    P5 is a degree-5 polynomial (6 coefficients).
    Q3 is a degree-3 polynomial (4 coefficients).
    Total parameters: 10.
    Product is a degree-8 polynomial.
    
    This is a different factorization structure than the previous P4*Q4.
    """
    # P5: deg 5, coeffs p0..p5
    p = params[0:6]
    # Q3: deg 3, coeffs q0..q3
    q = params[6:10]
    
    # Compute P(x) using Horner's method
    # P(x) = p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4 + p5*x^5
    p0, p1, p2, p3, p4, p5 = p
    Px = p5 * x**4 + p4 * x**3 + p3 * x**2 + p2 * x + p1
    Px = Px * x + p0
    
    # Compute Q(x) using Horner's method
    # Q(x) = q0 + q1*x + q2*x^2 + q3*x^3
    q0, q1, q2, q3 = q
    Qx = q3 * x**2 + q2 * x + q1
    Qx = Qx * x + q0
    
    return Px * Qx
# EVOLVE-BLOCK-END
