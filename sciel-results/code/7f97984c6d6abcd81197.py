"""Universal approximator strategy: linear combination of diverse basis functions."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a linear combination of diverse basis functions.
    The evaluator fits `params` (length 10) via BFGS.
    
    Basis functions chosen to cover common functional forms:
    1. Constant
    2. x
    3. y
    4. z
    5. x*y
    6. x*z
    7. y*z
    8. exp(x)
    9. exp(y)
    10. exp(z)
    
    This allows the model to represent additive separable functions, interactions,
    and exponential growth/decay, which are common in symbolic regression benchmarks.
    The linearity in params ensures stable optimization with BFGS.
    """
    # Ensure inputs are float arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    # Define basis functions
    # We use 10 specific basis functions to match the 10 parameters.
    # This is a "sparse" universal approximator where we pick the most likely components.
    
    # Basis 0: Constant
    b0 = np.ones_like(x)
    
    # Basis 1-3: Linear terms
    b1 = x
    b2 = y
    b3 = z
    
    # Basis 4-6: Pairwise interactions
    b4 = x * y
    b5 = x * z
    b6 = y * z
    
    # Basis 7-9: Exponential terms (capped to prevent overflow in optimization)
    # exp(x) for x in [0.5, 2] is in [1.6, 7.4], safe.
    b7 = np.exp(x)
    b8 = np.exp(y)
    b9 = np.exp(z)
    
    # Combine with parameters
    # params[0]*b0 + ... + params[9]*b9
    prediction = (params[0] * b0 +
                  params[1] * b1 +
                  params[2] * b2 +
                  params[3] * b3 +
                  params[4] * b4 +
                  params[5] * b5 +
                  params[6] * b6 +
                  params[7] * b7 +
                  params[8] * b8 +
                  params[9] * b9)
    
    # Ensure finite output
    if not np.all(np.isfinite(prediction)):
        return np.full_like(x, np.nan)
        
    return prediction
# EVOLVE-BLOCK-END
