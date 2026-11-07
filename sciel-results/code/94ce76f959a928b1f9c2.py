"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression via a small Parametric Neural Network (MLP).
    
    The 10 parameters are interpreted as:
    - params[0:6]: Weights from input layer (x, y) to 3 hidden units.
      Layout: [w11, w12, w13, w21, w22, w23]
      where w_ij is weight from input j (0=x, 1=y) to hidden unit i (0,1,2)
    - params[6:9]: Biases for the 3 hidden units.
    - params[9]:   Bias for the output unit.
    
    Architecture:
    Input: [x, y]
    Hidden: 3 units with tanh activation
    Output: 1 unit (linear)
    
    This allows fitting non-linear functions like sin(x), exp(y), x*y, etc.,
    which are difficult for simple linear basis functions.
    """
    start_time = time.time()
    
    # Ensure inputs are 1D arrays
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    n = len(x)
    
    if n == 0:
        return np.array([])
    
    # Safety margin: stop searching if we take too long
    # Although this function is just forward pass, the evaluator calls BFGS.
    # We just need to be fast in forward pass.
    DEADLINE = 140.0 

    # Check if we have enough params
    if len(params) < 10:
        # Fallback to linear if params are missing (shouldn't happen)
        p0 = params[0] if len(params) > 0 else 1.0
        p1 = params[1] if len(params) > 1 else 1.0
        p2 = params[2] if len(params) > 2 else 1.0
        return p0 * x + p1 * y + p2

    # Extract weights
    # Input to Hidden Weights (3 hidden units, 2 inputs)
    # Shape: (2, 3)
    W_in = np.array([
        [params[0], params[1], params[2]],  # Weights from x to h0, h1, h2
        [params[3], params[4], params[5]]   # Weights from y to h0, h1, h2
    ])
    
    # Hidden Biases (3)
    b_hidden = params[6:9]
    
    # Output Bias (1)
    # Note: We are using a linear output layer, so we only need the bias.
    # The weights from hidden to output are fixed to 1? 
    # No, that would be too restrictive. 
    # Wait, I have 10 params.
    # If I use 3 hidden units:
    # Weights in: 2 * 3 = 6
    # Biases in: 3
    # Weights out: 3 * 1 = 3
    # Bias out: 1
    # Total: 13 params. I only have 10.
    
    # Revised Architecture for 10 params:
    # Option A: 2 hidden units.
    # Weights in: 2 * 2 = 4
    # Biases in: 2
    # Weights out: 2 * 1 = 2
    # Bias out: 1
    # Total: 9 params. (1 spare)
    
    # Option B: 3 hidden units, but fix output weights to 1? 
    # If output weights are 1, the function is sum(tanh(...)) + b_out.
    # This is a common "sum of sigmoids/tanh" approximation.
    # Let's try Option B as it uses all 10 params and allows more flexibility in the hidden layer.
    # W_in: 6 params
    # b_hidden: 3 params
    # b_out: 1 param
    # Total: 10 params.
    
    # So:
    # W_in is (2, 3)
    # b_hidden is (3,)
    # Output is sum(hidden) + b_out
    
    # Let's re-verify the mapping:
    # params[0:6] -> W_in (flattened 2x3)
    # params[6:9] -> b_hidden
    # params[9]   -> b_out
    
    # Construct Input Matrix
    # X is (n, 2)
    X = np.column_stack((x, y))
    
    # Forward Pass
    # Z_in = X @ W_in.T + b_hidden
    # W_in is (2,3), so W_in.T is (3,2). X is (n,2). X @ W_in.T is (n,3).
    # b_hidden is (3,). Broadcasting works.
    
    try:
        Z_in = X @ W_in.T + b_hidden
        H = np.tanh(Z_in)
        
        # Output is sum of hidden units + bias
        # H is (n, 3). Sum over axis 1 gives (n,)
        pred = np.sum(H, axis=1) + params[9]
        
        # Check for non-finite values
        if not np.all(np.isfinite(pred)):
             # Fallback to simple linear if explosion occurs
             return params[0] * x + params[1] * y + params[2]
             
    except Exception:
        # Fallback to simple linear
        return params[0] * x + params[1] * y + params[2]
        
    return pred
# EVOLVE-BLOCK-END
