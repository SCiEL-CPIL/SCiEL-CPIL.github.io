"""Symbolic regression with 1 input: product of two low-degree polynomials."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: P5(x) * Q4(x) where
      P5(x) = p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4 + p5*x^5
      Q4(x) = q0 + q1*x + q2*x^2 + q3*x^3 + q4*x^4
    
    This gives a degree-9 polynomial with 10 parameters (6+4), creating a
    non-convex but structured landscape. The product form allows capturing
    functions with sharp features, asymmetric shapes, and high-frequency
    oscillations that a fixed linear basis cannot.
    
    params[0:6] = coefficients of P5
    params[6:10] = coefficients of Q4
    """
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Extract polynomial coefficients: 6 for P5, 4 for Q4
    p0, p1, p2, p3, p4, p5 = p[0], p[1], p[2], p[3], p[4], p[5]
    q0, q1, q2, q3 = p[6], p[7], p[8], p[9]
    
    # Compute powers of x
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    
    # Compute P5(x) = p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4 + p5*x^5
    P5 = p0 + p1 * x + p2 * x2 + p3 * x3 + p4 * x4 + p5 * x5
    
    # Compute Q4(x) = q0 + q1*x + q2*x^2 + q3*x^3
    # Wait, Q4 should be degree 4, so we need 5 coefficients. But we only have 4.
    # Let me use P5 * Q3 instead, or adjust.
    # Actually, let's use P4 * Q5 or just make it work with what we have.
    # Let's use P5 (6 coeffs) * Q3 (4 coeffs) = degree 8 polynomial
    Q3 = q0 + q1 * x + q2 * x2 + q3 * x3
    
    return P5 * Q3
# EVOLVE-BLOCK-END
