"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Chebyshev polynomial basis of degree 9.
    
    The input x is scaled to [-1, 1] to improve numerical stability.
    The function is represented as a linear combination of Chebyshev polynomials.
    This model is linear in parameters, ensuring the optimization landscape 
    is convex, allowing BFGS to find the global minimum reliably.
    """
    # Scale x from [0.1, 6] to approximately [-1, 1]
    # Center: (0.1 + 6) / 2 = 3.05
    # Range: 6 - 0.1 = 5.9, Half-range: 2.95
    # x_scaled = (x - 3.05) / 2.95
    
    x_scaled = (x - 3.05) / 2.95
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(x) = 1
    # T_1(x) = x
    # T_{n+1}(x) = 2*x*T_n(x) - T_{n-1}(x)
    
    # Initialize basis matrix: 10 columns for T_0...T_9
    # x is 1D array of shape (N,)
    # We want result shape (N, 10)
    
    # Ensure x is 1D for consistent indexing
    x_flat = np.asarray(x).flatten()
    N = len(x_flat)
    
    # If x is already 1D, this is fine. If 2D, we flatten and reshape back at end.
    # The task says "one prediction per point", implying 1D input/output.
    
    T = np.zeros((N, 10))
    
    if N == 0:
        return np.array([])
        
    T[:, 0] = 1.0
    if N > 0:
        T[:, 1] = x_scaled
        
    for i in range(2, 10):
        T[:, i] = 2.0 * x_scaled * T[:, i-1] - T[:, i-2]
        
    # Linear combination: sum(T[:, i] * params[i])
    # params is shape (10,)
    # Result is shape (N,)
    y_pred = np.dot(T, params)
    
    # Reshape to match input shape if necessary, though 1D is expected
    return y_pred.reshape(np.asarray(x).shape)
# EVOLVE-BLOCK-END
