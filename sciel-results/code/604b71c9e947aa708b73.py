import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    
    # Fixed random-like weights and biases for 20 hidden units
    # Using trigonometric functions for deterministic, varied values
    i = np.arange(20)
    w = np.cos(2 * np.pi * i / 20)
    b = np.sin(2 * np.pi * i / 20)
    
    # Compute hidden layer activations: ReLU(x * w + b)
    # x is (N,), w is (20,), b is (20,)
    # Need to broadcast x to (N, 1)
    z = x[:, np.newaxis] * w[np.newaxis, :] + b[np.newaxis, :]
    h = np.maximum(0, z)
    
    # Output weights: map 10 params to 20 weights
    # v[i] = params[i % 10]
    v = np.tile(params, 2) # This makes v[i] = params[i % 10]
    
    # y = h @ v
    y = h @ v
    
    return y
