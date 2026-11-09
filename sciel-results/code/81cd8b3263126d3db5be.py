"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Use a rich fixed basis: Polynomials, Trigonometric, and Tanh.
    # This makes the model linear in params, ensuring convex optimization for BFGS.
    # Basis functions:
    # 1. 1 (bias)
    # 2. x
    # 3. x^2
    # 4. x^3
    # 5. x^4
    # 6. x^5
    # 7. sin(pi*x)
    # 8. cos(pi*x)
    # 9. tanh(x)
    # 10. tanh(2*x)
    
    # Ensure x is a 1D array for consistent operations
    x = x.reshape(-1)
    
    # Compute basis matrix
    # Shape: (N, 10) where N is number of points
    basis = np.column_stack((
        np.ones_like(x),
        x,
        x**2,
        x**3,
        x**4,
        x**5,
        np.sin(np.pi * x),
        np.cos(np.pi * x),
        np.tanh(x),
        np.tanh(2 * x)
    ))
    
    # Linear combination
    y = basis @ params
    
    return y
# EVOLVE-BLOCK-END
