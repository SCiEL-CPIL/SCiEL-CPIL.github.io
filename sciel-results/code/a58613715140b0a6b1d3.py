"""Symbolic regression with 1 input using a fixed ReLU network with 20 hidden units."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict y from x using a small neural network structure.
    
    Architecture: 1 input -> 20 ReLU hidden units -> 1 output
    The 10 parameters control:
    - params[0..4]: output weights for 5 groups of hidden units
    - params[5]: global bias
    - params[6]: input scaling
    - params[7]: input shift  
    - params[8]: linear trend coefficient
    - params[9]: quadratic trend coefficient
    
    Hidden units have fixed centers spread across [-3, 3] with fixed widths.
    ReLU activation: max(0, x)
    """
    x = np.asarray(x, dtype=float)
    if x.ndim > 1:
        x = x.ravel()
    
    # Input transformation
    a = params[6]  # scaling
    b = params[7]  # shift
    x_scaled = a * x + b
    
    # 20 hidden units with fixed centers and widths
    # Centers spread evenly across [-3, 3]
    centers = np.linspace(-3.0, 3.0, 20)
    # Fixed widths for ReLU-like localized responses
    # Using a soft ReLU approximation: max(0, (x - c) * width)
    widths = np.full(20, 1.0)
    
    # Compute hidden layer activations: ReLU(x_scaled - center_i) * width_i
    # Shape: (n_points, 20)
    hidden = np.maximum(0.0, x_scaled[:, np.newaxis] - centers[np.newaxis, :]) * widths[np.newaxis, :]
    
    # Output weights: 5 groups of 4 hidden units each, each group shares one weight
    # params[0..4] are the 5 group weights
    group_weights = params[0:5]
    # Each group of 4 units: units 0-3 use weight 0, units 4-7 use weight 1, etc.
    # Reshape hidden to (n_points, 5, 4) and sum over last axis, then weight
    hidden_groups = hidden.reshape(hidden.shape[0], 5, 4)
    hidden_sum = hidden_groups.sum(axis=2)  # (n_points, 5)
    
    # Weighted sum of hidden group outputs
    y_nn = hidden_sum @ group_weights  # (n_points,)
    
    # Global trend terms
    y_trend = params[5] + params[8] * x + params[9] * x ** 2
    
    result = y_nn + y_trend
    
    return result
# EVOLVE-BLOCK-END
