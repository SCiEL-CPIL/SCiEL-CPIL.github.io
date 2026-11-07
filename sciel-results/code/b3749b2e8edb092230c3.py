import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: Combines full cubic polynomial (degree 3) with a fixed non-linear
    neural network basis. This provides both the structural expressiveness of the
    polynomial family and the non-linear flexibility of the neural network,
    while remaining linear in the 10 parameters for fast BFGS convergence.
    
    Parameters:
    x: array of shape (N,)
    y: array of shape (N,)
    params: array of shape (10,)
    
    Returns:
    array of shape (N,)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure x and y are 1D
    x = x.flatten()
    y = y.flatten()
    
    # Compute polynomial basis (10 terms, degree 3)
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x * x
    b4 = x * y
    b5 = y * y
    b6 = x * x * x
    b7 = x * x * y
    b8 = x * y * y
    b9 = y * y * y
    
    # Compute neural network basis (10 features)
    # Fixed random network weights for the basis
    # Using a fixed seed ensures reproducibility and identical basis across evaluations.
    # The basis is: h(x, y) = tanh(W2 @ tanh(W1 @ [x, y] + b1) + b2)
    
    rng = np.random.RandomState(12345)
    # First layer: 2 inputs -> 16 hidden units
    W1 = rng.randn(2, 16) * 1.0
    b1_nn = rng.randn(16) * 0.5
    
    # Second layer: 16 hidden -> 10 hidden (to match param length)
    W2 = rng.randn(16, 10) * 1.0
    b2_nn = rng.randn(10) * 0.5
    
    # Input matrix of shape (N, 2)
    X = np.column_stack((x, y))
    
    # Forward pass through the fixed network
    # Layer 1: (N, 2) @ (2, 16) -> (N, 16)
    Z1 = X @ W1 + b1_nn
    H1 = np.tanh(Z1)
    
    # Layer 2: (N, 16) @ (16, 10) -> (N, 10)
    Z2 = H1 @ W2 + b2_nn
    H2 = np.tanh(Z2)
    
    # Combine polynomial and neural network basis
    # We need to map the 10 params to a combination of these 20 features.
    # Strategy: Use the first 5 params for polynomial and last 5 for NN, or
    # better: Create a combined basis matrix of shape (N, 10) by mixing them.
    
    # Let's use a weighted combination:
    # pred = sum(p[i] * (poly_basis[i] + nn_basis[i]))
    # This effectively creates 10 hybrid features.
    
    # Polynomial basis as a matrix (N, 10)
    P = np.column_stack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))
    
    # Neural network basis is already (N, 10)
    
    # Combine: Hybrid basis = P + H2
    # This allows the parameters to fit both polynomial and non-linear components simultaneously.
    # If the function is purely polynomial, H2 terms will be fit with near-zero weights (or cancel out).
    # If the function has non-linear structure, the H2 components will capture it.
    
    Hybrid = P + H2
    
    # Linear combination
    pred = Hybrid @ p
    
    # Safety check
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
