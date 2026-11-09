"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """E. coli growth rate using a single-layer ReLU neural network.
    
    The model uses a hidden layer with fixed (non-trainable) basis functions
    and a linear output layer with trainable weights. This keeps the problem
    linear in the parameters, making the loss landscape convex for BFGS.
    
    Hidden units:
      1. b
      2. s
      3. temp
      4. pH
      5. b*s
      6. b*temp
      7. b*pH
      8. s*temp
      9. s*pH
      10. temp*pH
      11. 1 (bias)
    
    Output: sum of ReLU(hidden_unit * w_i) + bias
    """
    # Ensure inputs are float arrays
    b = np.asarray(b, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    pH = np.asarray(pH, dtype=np.float64)
    
    # Build hidden layer activations (fixed basis)
    # We have 10 parameters: 9 weights for hidden units + 1 bias
    # Hidden units are fixed functions of the inputs
    
    h1 = b
    h2 = s
    h3 = temp
    h4 = pH
    h5 = b * s
    h6 = b * temp
    h7 = b * pH
    h8 = s * temp
    h9 = s * pH
    h10 = temp * pH
    
    # Linear combination with ReLU activations
    # params[0] through params[8] are weights for h1-h9
    # params[9] is the bias
    
    w1, w2, w3, w4, w5, w6, w7, w8, w9, bias = params
    
    out = (np.maximum(w1 * h1, 0) +
           np.maximum(w2 * h2, 0) +
           np.maximum(w3 * h3, 0) +
           np.maximum(w4 * h4, 0) +
           np.maximum(w5 * h5, 0) +
           np.maximum(w6 * h6, 0) +
           np.maximum(w7 * h7, 0) +
           np.maximum(w8 * h8, 0) +
           np.maximum(w9 * h9, 0) +
           bias)
    
    # Ensure finite output
    return np.where(np.isfinite(out), out, 0.0)
# EVOLVE-BLOCK-END
