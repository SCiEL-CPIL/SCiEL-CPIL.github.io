# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with 10 parameters for symbolic regression.
    
    Approach: General Rational Function.
    This approach models the target function as a ratio of two polynomials.
    This differs from the seed program's additive basis by allowing for
    non-additive interactions and saturation effects.
    
    Model:
    Numerator: p0 + p1*x + p2*y + p3*z + p4*x*y + p5*y*z + p6*z*x
    Denominator: 1 + p7*x + p8*y + p9*z
    
    Parameters:
    p0: Bias in numerator
    p1-p3: Linear coefficients in numerator
    p4-p6: Pairwise interaction coefficients in numerator
    p7-p9: Linear coefficients in denominator (denom constant fixed to 1)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Calculate numerator
    # We switch from pairwise interactions to squared terms to better capture
    # separable non-linearities (curvature) in each dimension, which is a
    # common structure in symbolic regression benchmarks.
    num = (
        p0 +
        p1 * x +
        p2 * y +
        p3 * z +
        p4 * x**2 +
        p5 * y**2 +
        p6 * z**2
    )
    
    # Calculate denominator
    # Adding 1.0 ensures the denominator is not zero if params are small/positive
    # and helps with identifiability (scale of num/denom)
    den = (
        1.0 +
        p7 * x +
        p8 * y +
        p9 * z
    )
    
    # Prevent division by zero or near-zero by clipping denominator
    # Inputs are in [0.5, 2]. If params are reasonable, den should be safe.
    # But for robustness against BFGS wandering into bad regions:
    den_safe = np.where(np.abs(den) < 1e-6, 1e-6 * np.sign(den), den)
    # If den is exactly 0, sign is 0, so we use 1e-6.
    # Actually, np.sign(0) is 0. So 1e-6 * 0 is 0.
    # Better:
    den_safe = np.where(np.abs(den) < 1e-6, 1e-6, den)
    
    pred = num / den_safe
    
    return pred
# EVOLVE-BLOCK-END
