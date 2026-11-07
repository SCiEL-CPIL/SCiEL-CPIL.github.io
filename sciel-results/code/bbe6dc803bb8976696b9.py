"""Single-hidden-layer neural network with tanh activation for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Single-hidden-layer neural network (1 hidden neuron, tanh activation).
    
    Architecture:
      Input: [x, y, z, 1]  (4-dim with bias)
      Hidden: h = tanh(W1 @ input + b1)   -> scalar per point
      Output: f = w2_h * h + w2_x * x + w2_y * y + w2_z * z + b2
    
    Parameters (length 10):
      params[0]  = W1_x   (input weight for x to hidden)
      params[1]  = W1_y   (input weight for y to hidden)
      params[2]  = W1_z   (input weight for z to hidden)
      params[3]  = b1     (bias to hidden)
      params[4]  = w2_h   (hidden-to-output weight)
      params[5]  = w2_x   (direct x-to-output weight)
      params[6]  = w2_y   (direct y-to-output weight)
      params[7]  = w2_z   (direct z-to-output weight)
      params[8]  = b2     (output bias)
      params[9]  = scale  (global scale multiplier, softplus-transformed)
    
    The direct linear path (params[5:9]) ensures the model can always represent
    a linear function exactly, while the tanh hidden unit adds non-linear
    capacity. The scale parameter (softplus) provides an additional degree of
    freedom for amplitude.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    params = np.asarray(params, dtype=np.float64).ravel()
    if params.size != 10:
        params = np.ones(10)
    
    # Extract parameters
    W1_x = params[0]
    W1_y = params[1]
    W1_z = params[2]
    b1   = params[3]
    
    w2_h = params[4]
    w2_x = params[5]
    w2_y = params[6]
    w2_z = params[7]
    b2   = params[8]
    
    # Scale: softplus to keep positive, clamped for stability
    scale_raw = params[9]
    scale_abs = np.abs(scale_raw)
    scale = np.maximum(scale_raw, 0.0) + np.log1p(np.exp(-scale_abs))
    scale = np.clip(scale, 0.01, 100.0)
    
    # Compute hidden layer: tanh(W1_x * x + W1_y * y + W1_z * z + b1)
    hidden_input = W1_x * x + W1_y * y + W1_z * z + b1
    # Clamp hidden input to avoid numerical issues with tanh (though tanh is stable)
    hidden_input = np.clip(hidden_input, -50.0, 50.0)
    hidden = np.tanh(hidden_input)
    
    # Output: linear combination of hidden unit and direct inputs
    prediction = w2_h * hidden + w2_x * x + w2_y * y + w2_z * z + b2
    
    # Apply global scale
    prediction = scale * prediction
    
    # Ensure finite output
    if not np.all(np.isfinite(prediction)):
        return np.full_like(x, np.nan)
    
    return prediction
# EVOLVE-BLOCK-END
