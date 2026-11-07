"""
Symbolic regression using a linear combination of diverse basis functions.
This approach models y as a linear sum of 10 carefully chosen basis functions.
Since the model is linear in the parameters, the optimization problem is convex,
ensuring that scipy.optimize.minimize (BFGS) will reliably converge to the global minimum.
This avoids the local minima and instability issues associated with rational functions.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    x: array of input points
    params: array of 10 coefficients
    Returns: predicted values
    
    Basis functions:
    0: 1
    1: x
    2: x^2
    3: x^3
    4: x^4
    5: sin(x)
    6: cos(x)
    7: tanh(x)
    8: exp(-x^2)
    9: sin(2x)
    """
    # Compute basis functions
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    # Basis vector construction
    # Using a list of arrays and then stacking them for efficiency
    # However, for small N (200 points), direct computation is fine.
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = x2
    b3 = x3
    b4 = x4
    b5 = np.sin(x)
    b6 = np.cos(x)
    b7 = np.tanh(x)
    b8 = np.exp(-x2)
    b9 = np.sin(2 * x)
    
    # Construct the design matrix (N x 10)
    # We can do this efficiently without explicit loop overhead for large N,
    # but N=200 is small. Stacking is clear.
    # Alternatively, compute the weighted sum directly to save memory/time slightly.
    
    # Prediction = sum(params[i] * b_i)
    pred = (params[0] * b0 + 
            params[1] * b1 + 
            params[2] * b2 + 
            params[3] * b3 + 
            params[4] * b4 + 
            params[5] * b5 + 
            params[6] * b6 + 
            params[7] * b7 + 
            params[8] * b8 + 
            params[9] * b9)
            
    return pred
# EVOLVE-BLOCK-END
