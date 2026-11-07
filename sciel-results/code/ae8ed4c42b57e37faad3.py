import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluate a linear combination of 10 diverse basis functions.
    
    x: input array (values in [0.1, 6])
    params: array of length 10, coefficients for the basis functions
    
    Returns: array of predictions, one per point in x.
    """
    # Ensure x is a 1D array for consistent shape, though evaluator handles this
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    
    # Initialize basis matrix B of shape (n, 10)
    B = np.zeros((n, 10), dtype=np.float64)
    
    # 1. Constant
    B[:, 0] = 1.0
    
    # 2. Linear
    B[:, 1] = x
    
    # 3. Quadratic
    B[:, 2] = x * x
    
    # 4. Cubic
    B[:, 3] = x * x * x
    
    # 5. Exponential decay (slow)
    B[:, 4] = np.exp(-0.5 * x)
    
    # 6. Exponential decay (fast)
    B[:, 5] = np.exp(-1.5 * x)
    
    # 7. Rational
    B[:, 6] = 1.0 / (x + 0.5)
    
    # 8. Square root
    B[:, 7] = np.sqrt(x)
    
    # 9. Logarithm
    # x is in [0.1, 6], so log(x) is safe and finite
    B[:, 8] = np.log(x)
    
    # 10. Sine (low frequency)
    B[:, 9] = np.sin(x)
    
    # Compute predictions: B @ params
    return B @ params
