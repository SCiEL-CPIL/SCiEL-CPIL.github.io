"""Neural network with single hidden layer for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Single hidden layer neural network with 4 hidden units.
    
    params layout (length 10):
      - params[0:4]: weights from input to hidden layer (4 units)
      - params[4:8]: biases for hidden layer (4 units)
      - params[8:10]: output weights for 2 hidden units? 
      
    Wait, I need exactly 10 params. Let me design carefully.
    
    Let's use 3 hidden units with tanh activation:
      - 3 input-to-hidden weights
      - 3 hidden biases  
      - 3 hidden-to-output weights
      - 1 output bias
    Total: 3 + 3 + 3 + 1 = 10
    
    Architecture:
      hidden_i = tanh(w_i * x + b_i)  for i in 0..2
      output = sum_j (v_j * hidden_j) + c
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    w = params[0:3]      # input to hidden weights
    b = params[3:6]      # hidden biases
    v = params[6:9]      # hidden to output weights
    c = params[9]        # output bias
    
    # Compute hidden layer
    # x shape: (n,), w shape: (3,)
    # We need outer product: (n, 1) * (1, 3) -> (n, 3)
    z = np.outer(x, w) + b  # (n, 3)
    h = np.tanh(z)           # (n, 3)
    
    # Output layer
    out = h @ v + c          # (n,)
    
    return out
# EVOLVE-BLOCK-END
