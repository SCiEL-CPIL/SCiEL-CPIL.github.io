import numpy as np
from scipy.optimize import minimize

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a rational function with quadratic denominator.
    y = (p0 + p1*x + p2*v + p3*x^2 + p4*v^2 + p5*x*v + p6*x^3 + p7*v^3) / (1 + p8*x^2 + p9*v^2)
    
    This form can model saturation effects and bounded dynamics better than pure polynomials.
    Inputs are scaled to improve BFGS conditioning.
    """
    # Scale inputs to roughly [-1, 1]
    X = x / 0.6
    V = v / 0.3
    
    # Compute powers
    X2 = X * X
    X3 = X2 * X
    V2 = V * V
    V3 = V2 * V
    
    # Cross term
    XV = X * V
    
    # Numerator terms
    n0 = np.ones_like(x)
    n1 = X
    n2 = V
    n3 = X2
    n4 = V2
    n5 = XV
    n6 = X3
    n7 = V3
    
    # Denominator
    d = 1.0 + params[8] * X2 + params[9] * V2
    
    # Compute numerator
    numerator = (params[0] * n0 + params[1] * n1 + params[2] * n2 + 
                 params[3] * n3 + params[4] * n4 + params[5] * n5 +
                 params[6] * n6 + params[7] * n7)
    
    # Avoid division by zero or near-zero denominator
    # Add small epsilon for numerical stability
    epsilon = 1e-8
    safe_d = np.where(np.abs(d) < epsilon, np.sign(d) * epsilon + (d == 0) * epsilon, d)
    
    return numerator / safe_d
