# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Single-hidden-layer neural network with tanh activation.
    
    Architecture:
    - Input: [1, x, y, z] (4 features including bias)
    - Hidden layer: 3 neurons with tanh activation
    - Output: single linear combination
    
    Parameters (10 total):
    0-2:   Hidden layer weights (3x4 matrix flattened: w[0..3] for neuron 0, w[4..7] for neuron 1, w[8..11] for neuron 2)
           Wait, that's 12 params for weights alone. Let me recalculate.
    
    Actually with 3 hidden neurons and 4 inputs:
    - Hidden weights: 3 * 4 = 12 params
    - Hidden biases: 3 params
    - Output weights: 3 params
    - Output bias: 1 param
    Total: 19 params, but we only have 10.
    
    Let me use 2 hidden neurons instead:
    - Hidden weights: 2 * 4 = 8 params
    - Hidden biases: 2 params
    - Output weights: 2 params
    - Output bias: 1 param
    Total: 13 params. Still too many.
    
    Let me try 1 hidden neuron:
    - Hidden weights: 1 * 4 = 4 params
    - Hidden bias: 1 param
    - Output weight: 1 param
    - Output bias: 1 param
    Total: 7 params. We have 3 extra.
    
    Let me use 2 hidden neurons but share some structure:
    Actually, let me just use exactly 10 params:
    - 2 hidden neurons
    - Each hidden neuron gets weights for [x, y, z] (3 params each = 6)
    - 2 hidden biases (2 params)
    - 2 output weights (2 params)
    - 1 output bias (1 param)
    Total: 6 + 2 + 2 + 1 = 11. Still one too many.
    
    Let me drop the output bias and use:
    - 2 hidden neurons
    - Each hidden neuron gets weights for [x, y, z] (3 params each = 6)
    - 2 hidden biases (2 params)
    - 2 output weights (2 params)
    Total: 10 params.
    
    Architecture:
    h1 = tanh(p[0]*x + p[1]*y + p[2]*z + p[3])
    h2 = tanh(p[4]*x + p[5]*y + p[6]*z + p[7])
    out = p[8]*h1 + p[9]*h2
    """
    p = params
    
    # Hidden layer with tanh activation
    h1 = np.tanh(p[0] * x + p[1] * y + p[2] * z + p[3])
    h2 = np.tanh(p[4] * x + p[5] * y + p[6] * z + p[7])
    
    # Output layer (linear combination of hidden activations)
    out = p[8] * h1 + p[9] * h2
    
    return out
# EVOLVE-BLOCK-END
