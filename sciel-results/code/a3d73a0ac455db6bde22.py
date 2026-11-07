import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a linear combination of 10 diverse basis functions.
    The basis is designed to capture polynomials, saturations, oscillations, and exponentials
    on the domain [-2, 2].
    
    Basis functions:
    0: 1
    1: x
    2: x^2
    3: x^3
    4: tanh(x)
    5: tanh(2*x)
    6: exp(-x^2)
    7: sin(x)
    8: cos(x)
    9: exp(-0.5*x)
    """
    # Ensure x is a flat array for consistent broadcasting if needed, 
    # though the contract implies 1D input.
    x = np.asarray(x, dtype=np.float64)
    
    # Compute basis functions
    # Using np.stack to create a matrix of shape (N, 10)
    # This is efficient for N=200.
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = x * x
    b3 = x * x * x
    b4 = np.tanh(x)
    b5 = np.tanh(2.0 * x)
    b6 = np.exp(-x * x)
    b7 = np.sin(x)
    b8 = np.cos(x)
    b9 = np.exp(-0.5 * x)
    
    # Stack into (N, 10) matrix
    # This creates a copy, but for N=200 it's negligible.
    basis = np.stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9], axis=1)
    
    # Compute linear combination: basis (N, 10) @ params (10,) -> (N,)
    # Using dot product is very fast.
    result = basis @ params
    
    return result
