"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid Global Basis Model for Symbolic Regression.
    
    This model uses a fixed set of 10 diverse basis functions that are linear
    in the parameters. The basis includes rational, trigonometric, exponential,
    and polynomial terms. This structural diversity allows the model to fit
    a wider class of hidden functions (e.g., f = 1/x + sin(y)) more accurately
    than pure polynomials or local RBFs, while maintaining linearity in params
    for robust optimization with BFGS.
    """
    # Ensure inputs are float arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Check for valid params
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
        
    # Define 10 diverse basis functions
    # 0: Constant
    f0 = 1.0
    
    # 1-3: Linear terms
    f1 = x
    f2 = y
    f3 = z
    
    # 4-6: Rational terms (common in symbolic regression, hard for polynomials)
    # Domain is [0.5, 2.0], so no division by zero.
    f4 = 1.0 / x
    f5 = 1.0 / y
    f6 = x / y  # or y/x, let's pick x/y for variety
    
    # 7-8: Trigonometric terms (capture periodicity)
    f7 = np.sin(x)
    f8 = np.cos(y)
    
    # 9: Exponential term (capture decay/growth)
    f9 = np.exp(-z)
    
    # Construct prediction as linear combination
    # Using np.dot or sum of products for efficiency and clarity
    pred = (p[0] * f0 + 
            p[1] * f1 + 
            p[2] * f2 + 
            p[3] * f3 + 
            p[4] * f4 + 
            p[5] * f5 + 
            p[6] * f6 + 
            p[7] * f7 + 
            p[8] * f8 + 
            p[9] * f9)
            
    return pred
# EVOLVE-BLOCK-END
