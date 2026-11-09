"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a Tunable Rational Function Basis.
    
    The model is of the form:
    y(x) = offset + (a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4) / (1.0 + b1*x + b2*x^2 + b3*x^3 + b4*x^4)
    
    This allows the model to capture rational structures (e.g., hyperbolic, rational curves) 
    that global polynomials approximate poorly without high degrees.
    
    Parameters:
    x: Input array of shape (N,)
    params: Array of 10 parameters [a0, a1, a2, a3, a4, b1, b2, b3, b4, offset]
    """
    # Unpack parameters
    a0, a1, a2, a3, a4, b1, b2, b3, b4, offset = params
    
    # Compute Numerator: P(x)
    # Using Horner's method for stability and speed
    # P(x) = a0 + x*(a1 + x*(a2 + x*(a3 + x*a4)))
    num = a0 + x * (a1 + x * (a2 + x * (a3 + x * a4)))
    
    # Compute Denominator: Q(x)
    # Q(x) = 1.0 + x*(b1 + x*(b2 + x*(b3 + x*b4)))
    # The constant term is fixed to 1.0 to ensure Q(0) != 0 initially 
    # and provide a baseline scale. This helps stability during BFGS from all-ones start.
    den = 1.0 + x * (b1 + x * (b2 + x * (b3 + x * b4)))
    
    # Safety check: Avoid division by zero or near-zero.
    # If the denominator gets too close to zero, the function value explodes.
    # We clip the denominator to avoid inf/nan which would crash the evaluator.
    # A small epsilon is added to prevent exact zero.
    # To handle sign changes safely, we can clip the absolute value if it's too small,
    # but simply adding a small constant might bias.
    # Better: np.where(abs(den) < 1e-6, 1e-6, den)
    # However, if the true function has a pole, this is correct behavior (large values).
    # If the true function is smooth, BFGS will move the pole away.
    # We must ensure no NaNs.
    
    # Use a safe division
    # If den is very small, the result is huge. 
    # To prevent overflow, we can clip the denominator magnitude.
    # But standard practice is just to let it be, unless it's exactly 0.
    # np.divide with where clause or just adding 1e-9.
    # Adding 1e-9 is safer for gradient computation if it's exactly 0.
    
    den_safe = den + 1e-9
    
    result = offset + num / den_safe
    
    return result
# EVOLVE-BLOCK-END
