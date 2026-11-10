"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: P5(x) * Q4(x) where P5 is a degree-5 polynomial and Q4 is a degree-4 polynomial.
    params[0:5] -> coefficients of P5: p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4 + p5*x^5 (wait, 5 params for deg 5? No, deg 5 needs 6 coeffs. Let's adjust.)
    
    Actually, let's use:
    P(x) = params[0] + params[1]*x + params[2]*x^2 + params[3]*x^3 + params[4]*x^4 + params[5]*x^5  (6 params, deg 5)
    Q(x) = params[6] + params[7]*x + params[8]*x^2 + params[9]*x^3                         (4 params, deg 3)
    Total params: 10.
    Product is a degree 8 polynomial.
    
    Let's stick to the directive: P5(x) * Q4(x).
    P5 has 6 coefficients (deg 0 to 5).
    Q4 has 5 coefficients (deg 0 to 4).
    Total 11 params. We have 10.
    
    Let's use P4(x) * Q5(x)? Or P5(x) * Q3(x)?
    Directive says: "product of two low-degree polynomials in x (e.g., P5(x) * Q4(x)), where the 10 parameters define the coefficients of these polynomials".
    If P5 is deg 5 (6 coeffs) and Q4 is deg 4 (5 coeffs), that's 11.
    Maybe P5 means 5 terms? i.e. deg 4?
    Let's assume "P5" means 5 parameters (deg 4) and "Q4" means 4 parameters (deg 3).
    Total 9 params. We have 10.
    Let's use P5 (5 params, deg 4) and Q5 (5 params, deg 4). Total 10.
    Product is deg 8.
    
    Let's define:
    P(x) = p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4
    Q(x) = q0 + q1*x + q2*x^2 + q3*x^3 + q4*x^4
    y = P(x) * Q(x)
    
    params[0:5] are p0..p4
    params[5:10] are q0..q4
    """
    p = params[0:5]
    q = params[5:10]
    
    # Compute P(x)
    # Use Horner's method for stability
    p0, p1, p2, p3, p4 = p
    Px = p4 * x**3 + p3 * x**2 + p2 * x + p1
    Px = Px * x + p0
    
    # Compute Q(x)
    q0, q1, q2, q3, q4 = q
    Qx = q4 * x**3 + q3 * x**2 + q2 * x + q1
    Qx = Qx * x + q0
    
    return Px * Qx
# EVOLVE-BLOCK-END
