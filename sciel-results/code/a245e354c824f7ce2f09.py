import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod.
    Uses a rational function for non-linear hardening/saturation
    combined with linear elastic and temperature interaction terms.
    """
    eps = strain
    T = temp
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Elastic term with temperature dependence
    elastic_term = (p0 + p1 * T) * eps
    
    # Rational non-linear term
    # Numerator: quadratic and cubic strain terms
    # Denominator: constant + quadratic strain term
    # This allows for saturation or inflection points
    denom = 1.0 + p4 * eps**2
    # Add small epsilon to denominator for numerical stability
    denom = denom + 1e-8
    numerator = p2 * eps**2 + p3 * eps**3
    rational_term = numerator / denom
    
    # Bias terms (constant and linear in T)
    bias_term = p5 + p6 * T
    
    # Interaction terms
    interaction_term = p7 * eps * T + p8 * eps**2 * T + p9 * eps**3 * T
    
    return elastic_term + rational_term + bias_term + interaction_term
