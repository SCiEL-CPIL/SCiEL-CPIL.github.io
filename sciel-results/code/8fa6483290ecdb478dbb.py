import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fully differentiable neural network basis with 10 output weights as parameters.
    
    Architecture:
    - Input: (x, y)
    - Layer 1: 2 -> 16, tanh activation
    - Layer 2: 16 -> 10, tanh activation
    - Output: 10 -> 1, linear (weights are the 10 params)
    
    The internal weights are fixed (deterministic), so the model is linear in params.
    This allows BFGS to optimize the 10 output weights efficiently while the fixed
    non-linear transformations provide universal approximation capability.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    x = x.flatten()
    y = y.flatten()
    
    # Fixed random network weights for the basis
    rng = np.random.RandomState(42)
    
    # Layer 1: 2 inputs -> 16 hidden units
    W1 = rng.randn(2, 16) * 1.0
    b1 = rng.randn(16) * 0.5
    
    # Layer 2: 16 hidden -> 10 hidden
    W2 = rng.randn(16, 10) * 1.0
    b2 = rng.randn(10) * 0.5
    
    # Input matrix of shape (N, 2)
    X = np.column_stack((x, y))
    
    # Forward pass through the fixed network
    # Layer 1: (N, 2) @ (2, 16) -> (N, 16)
    Z1 = X @ W1 + b1
    H1 = np.tanh(Z1)
    
    # Layer 2: (N, 16) @ (16, 10) -> (N, 10)
    Z2 = H1 @ W2 + b2
    H2 = np.tanh(Z2)
    
    # Linear combination with the 10 params
    pred = H2 @ p
    
    # Safety check
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
