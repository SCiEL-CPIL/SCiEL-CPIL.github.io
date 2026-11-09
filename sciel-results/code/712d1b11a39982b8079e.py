"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Construct a fixed basis of 10 functions.
    # The model is linear in params, which ensures the optimization landscape
    # is convex (or at least well-behaved for BFGS) with respect to params.
    
    # Basis functions:
    # 0: 1 (bias)
    # 1: x
    # 2: x^2
    # 3: x^3
    # 4: x^4
    # 5: sin(x)
    # 6: cos(x)
    # 7: tanh(x)
    # 8: sin(2x)
    # 9: cos(2x)
    
    # Ensure x is a 1D array for consistent broadcasting
    x = np.asarray(x).flatten()
    
    # Build the feature matrix
    # Shape: (N, 10)
    features = np.column_stack((
        np.ones_like(x),
        x,
        x**2,
        x**3,
        x**4,
        np.sin(x),
        np.cos(x),
        np.tanh(x),
        np.sin(2 * x),
        np.cos(2 * x)
    ))
    
    # Linear combination of features
    return features @ params
# EVOLVE-BLOCK-END
