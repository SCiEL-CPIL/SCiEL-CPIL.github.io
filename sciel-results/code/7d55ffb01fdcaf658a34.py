# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fixed two-layer tanh network whose output-layer weights are the 10 optimizable params.
    The internal weights are fixed random matrices seeded so that BFGS sees a convex
    (linear-in-params) problem over a rich set of non-linear features.
    
    Architecture:
      input (x, y) -> 20 tanh units (fixed random weights) -> 20 tanh units (fixed random weights) -> 10 params -> prediction
    
    This gives a 10-dimensional linear model over 10 fixed non-linear basis functions
    (each basis function is a composition of two tanh layers), providing universal
    approximation capability while keeping the optimization convex in the free parameters.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    n = len(x)
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Inputs: (n, 2)
    inp = np.column_stack([x, y])
    
    # Fixed random weights for first hidden layer: (2, 20)
    # Use a fixed seed for reproducibility
    rng = np.random.RandomState(42)
    W1 = rng.randn(2, 20) * 1.0
    b1 = np.zeros(20)
    
    # First hidden layer: (n, 20)
    h1 = np.tanh(inp @ W1 + b1)
    
    # Fixed random weights for second hidden layer: (20, 20)
    W2 = rng.randn(20, 20) * 1.0
    b2 = np.zeros(20)
    
    # Second hidden layer: (n, 20)
    h2 = np.tanh(h1 @ W2 + b2)
    
    # Output layer: linear combination with the 10 optimizable params
    # We need to map from 20 features to 10 params. 
    # Actually, we want the prediction to be a scalar per point.
    # So the output should be h2 @ w_out where w_out has 20 elements, but we only have 10 params.
    # 
    # Re-thinking: The 10 params should be the output weights.
    # So we need 10 basis functions, each being a scalar per point.
    # Let's design 10 fixed non-linear feature functions, then pred = sum(p[i] * f[i](x,y))
    
    # Let me redesign: 10 basis functions, each a composition of tanh layers
    # Basis i: tanh(W2_i @ tanh(W1_i @ inp + b1_i) + b2_i)
    # where W1_i is (2, 10), W2_i is (10, 1)
    
    # This gives 10 scalar features, and pred = h_features @ p
    
    # Regenerate with proper structure
    W1_all = rng.randn(2, 10, 10)  # (in_dim=2, num_banks=10, hidden=10)
    b1_all = np.zeros((10, 10))     # (num_banks=10, hidden=10)
    W2_all = rng.randn(10, 10, 1)   # (hidden=10, num_banks=10, out=1)
    b2_all = np.zeros((10, 1))      # (num_banks=10, out=1)
    
    # For each bank i, compute: tanh(tanh(inp @ W1_all[:, i, :] + b1_all[i, :]) @ W2_all[:, i, :] + b2_all[i, :])
    # Result: (n, 10) matrix of features
    
    features = np.zeros((n, 10), dtype=np.float64)
    for i in range(10):
        h1_i = np.tanh(inp @ W1_all[:, i, :] + b1_all[i, :])  # (n, 10)
        h2_i = np.tanh(h1_i @ W2_all[:, i, :] + b2_all[i, :])  # (n, 1)
        features[:, i] = h2_i.ravel()
    
    pred = features @ p
    
    return pred
# EVOLVE-BLOCK-END
