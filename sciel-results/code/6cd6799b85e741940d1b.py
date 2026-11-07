# EVOLVE-BLOCK-START
import numpy as np
import time

# Single hidden layer neural network with 10 ReLU units.
# params = [w1, b1, w2, b2, w3, b3, w4, b4, w5, b5]
# where wi are weights (scalar) and bi are biases (scalar) for each hidden unit.
# Output is sum of ReLU(wi * x_norm + bi) for i=1..10, scaled by a fixed output weight.
# Actually, we need 10 params total. Let's use:
# 5 hidden units, each with 2 params: weight and bias.
# Output: sum of ReLU(wi * x + bi) for i=1..5, multiplied by a fixed scale.
# But that's only 10 params for 5 units. The output is linear combination with fixed weights?
# No, the task says params is an array of length 10. We need to use all 10.
# Let's use: 10 hidden units, but we need 20 params (10 weights + 10 biases).
# We only have 10 params. So we can't have 10 independent units with both weight and bias.
# Alternative: Use 5 hidden units with shared structure, or use a different decomposition.
# Let's use: params = [w1, w2, w3, w4, w5, b1, b2, b3, b4, b5]
# Output = sum_{i=1}^{5} ReLU(wi * x_norm + bi)
# This gives us 5 ReLU units with 10 parameters total.

_X_NORM_CENTER = 3.05
_X_NORM_SCALE = 2.95


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Entry point. Returns predictions from a 5-unit ReLU neural network.
    params = [w1, w2, w3, w4, w5, b1, b2, b3, b4, b5]
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if len(params) != 10:
        # Fallback: return zeros if params has wrong length
        return np.zeros_like(x)
    
    # Normalize x
    x_norm = (x - _X_NORM_CENTER) / _X_NORM_SCALE
    
    # Extract weights and biases
    w = params[0:5]
    b = params[5:10]
    
    # Compute hidden layer: ReLU(wi * x_norm + bi)
    # w is shape (5,), x_norm is shape (n,)
    # We need outer product: each wi multiplies all x_norm
    # hidden[i] = ReLU(w[i] * x_norm + b[i])
    # hidden shape: (5, n)
    hidden = np.tanh(w[:, np.newaxis] * x_norm[np.newaxis, :] + b[:, np.newaxis])
    
    # Output: sum of hidden units (fixed output weights of 1)
    pred = np.sum(hidden, axis=0)
    
    if not np.all(np.isfinite(pred)):
        return np.zeros_like(x)
    
    return pred
# EVOLVE-BLOCK-END
