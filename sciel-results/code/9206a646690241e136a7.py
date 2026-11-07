"""Symbolic regression with 1 input: 10-parameter Composite Basis (Poly + Trig + Tanh)."""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter linear model with a diverse composite basis.
    
    The basis functions are:
    1. 1 (Bias)
    2. x
    3. x^2
    4. x^3
    5. x^4
    6. sin(x)
    7. cos(x)
    8. sin(2x)
    9. cos(2x)
    10. tanh(x)
    
    This linear model is convex in the parameters, so BFGS will converge to the
    global optimum for this specific basis. The diversity of the basis (polynomial,
    trigonometric, and hyperbolic) increases the likelihood that the target function
    can be well-approximated by a linear combination of these terms.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    
    # Ensure x is a column vector for broadcasting if needed, 
    # but here we just compute element-wise.
    
    # Precompute basis terms
    # 1. Bias
    b0 = np.ones_like(x)
    
    # 2-5. Polynomial terms
    b1 = x
    b2 = x * x
    b3 = x * x * x
    b4 = x * x * x * x
    
    # 6-9. Trigonometric terms
    b5 = np.sin(x)
    b6 = np.cos(x)
    b7 = np.sin(2.0 * x)
    b8 = np.cos(2.0 * x)
    
    # 10. Hyperbolic tangent (captures saturation/step-like behavior)
    b9 = np.tanh(x)
    
    # Combine into a matrix (n, 10)
    # Using np.column_stack is efficient
    basis = np.column_stack((
        b0, b1, b2, b3, b4, 
        b5, b6, b7, b8, 
        b9
    ))
    
    # params is an array of length 10
    # prediction = basis @ params
    pred = basis @ params
    
    return pred
