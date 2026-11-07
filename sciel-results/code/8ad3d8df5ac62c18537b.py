"""
Strategy: Linear combination of a rich fixed symbolic basis.
This approach directly represents common symbolic regression components
(polynomials, interactions, trigonometric functions) which are often
hard for small neural networks to learn accurately with limited parameters.
"""
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure inputs are arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Pre-compute basis functions
    # 1. Constant
    b0 = np.ones_like(x)
    # 2. Linear
    b1 = x
    b2 = y
    # 3. Interaction
    b3 = x * y
    # 4. Quadratic
    b4 = x * x
    b5 = y * y
    # 5. Trigonometric (common in symbolic regression)
    b6 = np.sin(x)
    b7 = np.sin(y)
    b8 = np.cos(x)
    b9 = np.cos(y)
    
    # Linear combination of basis functions
    # This allows the optimizer to weight each component independently
    return (params[0] * b0 + 
            params[1] * b1 + 
            params[2] * b2 + 
            params[3] * b3 + 
            params[4] * b4 + 
            params[5] * b5 + 
            params[6] * b6 + 
            params[7] * b7 + 
            params[8] * b8 + 
            params[9] * b9)
