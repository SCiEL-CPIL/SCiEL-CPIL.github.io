"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

# Fixed random weights for the hidden layer to create non-linear basis functions.
# This creates a "Random Fourier Features" style model but with tanh.
# The model is linear in `params`, so BFGS can optimize it efficiently.
# We use a fixed seed to ensure reproducibility.
_RNG = np.random.RandomState(42)
# 10 hidden units, each taking a linear combination of x and y
_W1 = _RNG.randn(10, 2) * 1.5 # Scale to ensure tanh is not in saturation regime too often


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers a function using a fixed non-linear basis (tanh of random projections).
    
    The model is:
    f(x, y) = sum_{i=0}^{9} p_i * tanh(w_{i,0} * x + w_{i,1} * y)
    
    Where w_{i,j} are fixed random weights.
    This is linear in params, allowing efficient optimization via BFGS.
    The non-linearity allows approximating a wider class of functions than polynomials.
    """
    # Stack x and y into a matrix of shape (N, 2)
    # x and y are assumed to be 1D arrays of the same length
    inputs = np.column_stack((x, y))
    
    # Compute hidden layer activations: (N, 2) @ (2, 10) -> (N, 10)
    # _W1 is (10, 2), so we need to transpose it for standard matrix multiplication
    # Z = inputs @ _W1.T
    Z = inputs @ _W1.T
    H = np.tanh(Z)
    
    # Compute predictions: H (N, 10) @ params (10,) -> (N,)
    pred = H @ params
    
    # Ensure finite values (safety)
    if not np.all(np.isfinite(pred)):
        # Fallback to simple linear if something went wrong
        return params[0] * x + params[1] * y + params[2]
        
    return pred
# EVOLVE-BLOCK-END
