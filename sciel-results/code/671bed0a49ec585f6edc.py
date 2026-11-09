import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational Nonlinear Oscillator Model.
    
    Strategy:
    Instead of an additive polynomial basis, this model uses a rational function form:
    y = N(x, v) / D(x, v)
    
    This structure allows the model to capture non-linear coupling and saturation effects
    (e.g., transition from cubic to linear restoring force at large amplitudes) that are
    difficult to represent with a simple sum of polynomials.
    
    To ensure numerical stability and avoid singularities during BFGS optimization,
    the denominator is constructed as 1 + c8^2 * x^2 + c9^2 * v^2, which is always >= 1.
    
    Parameters:
    c0-c7: Coefficients for the numerator (linear, quadratic, cubic, cross terms, constant).
    c8, c9: Coefficients for the denominator (stiffness modulation).
    """
    p = params
    
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = p
    
    # Precompute powers
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    xv = x * v
    x2v = x2 * v
    xv2 = x * v2
    
    # Numerator: Generalized polynomial up to 3rd order
    # Includes constant, linear, quadratic, cubic, and cross terms
    numerator = (
        c0 * x + 
        c1 * v + 
        c2 * x3 + 
        c3 * v2 + 
        c4 * xv + 
        c5 * x2v + 
        c6 * xv2 + 
        c7
    )
    
    # Denominator: Safe positive-definite form
    # Using squares of parameters ensures D >= 1, preventing division by zero
    # and keeping the landscape well-conditioned for BFGS
    denominator = 1.0 + c8 * c8 * x2 + c9 * c9 * v2
    
    return numerator / denominator
