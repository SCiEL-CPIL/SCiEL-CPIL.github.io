"""Rewritten program with a mixed basis of polynomials and non-linear activations."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts y using a linear combination of 10 diverse basis functions.
    The basis is designed to capture linear, quadratic, cubic, periodic,
    exponential, and non-smooth features within the domain [-2, 2].
    """
    # Ensure x is a numpy array
    x = np.asarray(x, dtype=float)
    
    # Basis functions
    # 1. Constant
    b0 = np.ones_like(x)
    # 2. Linear
    b1 = x
    # 3. Quadratic
    b2 = x**2
    # 4. Cubic
    b3 = x**3
    # 5. Quartic
    b4 = x**4
    # 6. Sine
    b5 = np.sin(x)
    # 7. Cosine
    b8 = np.cos(x)
    # 8. Gaussian bump centered at 0
    b7 = np.exp(-x**2)
    # 9. Tanh (smooth step)
    b8_tanh = np.tanh(x)
    # 10. Absolute value (captures kinks)
    b9 = np.abs(x)
    
    # Construct the design matrix (N x 10)
    # Note: The order of params must match the order of basis functions here
    # params[0] * b0 + params[1] * b1 + ...
    
    # Using matrix multiplication for efficiency
    # basis_matrix shape: (10, N)
    basis_matrix = np.vstack([
        b0,
        b1,
        b2,
        b3,
        b4,
        b5,
        b7,
        b8,
        b8_tanh,
        b9
    ])
    
    # Prediction: params @ basis_matrix
    # params shape: (10,)
    # basis_matrix shape: (10, N)
    # Result shape: (N,)
    
    prediction = params @ basis_matrix
    
    return prediction
# EVOLVE-BLOCK-END
