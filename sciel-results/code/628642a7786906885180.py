import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a nonlinear feedforward network structure.
    
    The 10 parameters are the output layer weights (plus bias) of a 2-hidden-layer
    network with tanh activations. The hidden layers have fixed weights that are
    chosen to provide diverse, non-separable feature transformations.
    
    Architecture:
    - Input: [x, y]
    - Hidden layer 1: 4 neurons with tanh activation
      - Fixed weights provide diverse linear combinations of x, y, x*y, x^2-y^2
    - Hidden layer 2: 4 neurons with tanh activation  
      - Fixed weights mix hidden layer 1 outputs
    - Output layer: 10 parameters (9 weights + 1 bias)
    
    This allows the model to implicitly learn complex functional forms through
    the fixed internal nonlinearity, rather than relying on explicit polynomial
    or trigonometric terms.
    """
    start_time = time.time()
    
    # Ensure x and y are 1D arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Input layer: [x, y]
    # Hidden layer 1: 4 neurons
    # Fixed weights for hidden layer 1 (4x2 matrix)
    # Neuron 0: x
    # Neuron 1: y
    # Neuron 2: x*y
    # Neuron 3: x^2 - y^2 (captures radial structure)
    h1_0 = x
    h1_1 = y
    h1_2 = x * y
    h1_3 = x * x - y * y
    
    # Apply tanh activation to hidden layer 1
    a1_0 = np.tanh(h1_0)
    a1_1 = np.tanh(h1_1)
    a1_2 = np.tanh(h1_2)
    a1_3 = np.tanh(h1_3)
    
    # Hidden layer 2: 4 neurons
    # Fixed weights mixing hidden layer 1 outputs
    # Neuron 0: a1_0 + a1_1
    # Neuron 1: a1_0 - a1_1
    # Neuron 2: a1_2 + a1_3
    # Neuron 3: a1_2 - a1_3
    h2_0 = a1_0 + a1_1
    h2_1 = a1_0 - a1_1
    h2_2 = a1_2 + a1_3
    h2_3 = a1_2 - a1_3
    
    # Apply tanh activation to hidden layer 2
    a2_0 = np.tanh(h2_0)
    a2_1 = np.tanh(h2_1)
    a2_2 = np.tanh(h2_2)
    a2_3 = np.tanh(h2_3)
    
    # Output layer: linear combination of hidden layer 2 activations
    # 10 parameters: 4 weights for a2_0..a2_3, plus we need 6 more
    # Let's use a richer output: combine all hidden activations
    # Actually, let's use a better design:
    # Output = w0*a2_0 + w1*a2_1 + w2*a2_2 + w3*a2_3 + w4*a1_0 + w5*a1_1 + w6*a1_2 + w7*a1_3 + w8*(x*y) + w9*(x^2+y^2)
    
    # Feature vector for output layer (9 features + bias)
    # Features: a2_0, a2_1, a2_2, a2_3, a1_0, a1_1, a1_2, a1_3, x*y
    # Bias: 1
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 140.0:
        # Safety margin: return a simple linear prediction
        return params[0] * np.ones_like(x) + params[1] * x + params[2] * y
    
    # Compute output as linear combination
    # params[0..8] are weights for the 9 features, params[9] is bias
    result = (params[0] * a2_0 +
              params[1] * a2_1 +
              params[2] * a2_2 +
              params[3] * a2_3 +
              params[4] * a1_0 +
              params[5] * a1_1 +
              params[6] * a1_2 +
              params[7] * a1_3 +
              params[8] * (x * y) +
              params[9] * np.ones_like(x))
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        # Fallback to simple linear
        result = params[0] * np.ones_like(x) + params[1] * x + params[2] * y
    
    return result
