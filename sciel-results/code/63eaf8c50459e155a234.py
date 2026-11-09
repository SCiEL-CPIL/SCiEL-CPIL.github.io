"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Universal Approximator using a single hidden layer Neural Network structure.
    
    Architecture:
    Input x (scalar) -> Hidden Layer (10 neurons, Tanh activation) -> Output Layer (1 neuron, Linear)
    
    Parameters mapping (10 total):
    - params[0:8] : Weights from input to hidden layer (w_h1 ... w_h8) 
                    Note: We use 8 hidden units to keep param count low and stable, 
                    but actually we have 10 params total. 
                    Let's re-allocate: 
                    Hidden layer size K. 
                    Params: K weights (input->hidden), K biases (hidden), 
                    K output weights (hidden->output), 1 output bias.
                    Total = 2K + 1 + 1? No.
                    
    Let's use a simpler, robust structure that fits exactly 10 params:
    Model: y = sum_{i=1}^N w_i * tanh(v_i * x + b_i) + c
    
    If N=4:
    Params: v1, b1, w1, v2, b2, w2, v3, b3, w3, v4, b4, w4, c -> 13 params. Too many.
    
    If N=3:
    Params: v1, b1, w1, v2, b2, w2, v3, b3, w3, c -> 10 params. Perfect.
    
    Structure:
    y = w1*tanh(v1*x + b1) + w2*tanh(v2*x + b2) + w3*tanh(v3*x + b3) + c
    
    This is a sum of 3 sigmoidal/tanh basis functions with variable centers (via bias) 
    and scales (via slope v). This is a very flexible model for smooth functions 
    on [-3, 3] and often outperforms fixed polynomials or sines for unknown targets.
    
    params[0] = v1 (slope/scale)
    params[1] = b1 (bias/center)
    params[2] = w1 (amplitude)
    params[3] = v2 (slope/scale)
    params[4] = b2 (bias/center)
    params[5] = w2 (amplitude)
    params[6] = v3 (slope/scale)
    params[7] = b3 (bias/center)
    params[8] = w3 (amplitude)
    params[9] = c  (offset)
    """
    
    # Extract parameters
    v1, b1, w1 = params[0], params[1], params[2]
    v2, b2, w2 = params[3], params[4], params[5]
    v3, b3, w3 = params[6], params[7], params[8]
    c = params[9]
    
    # Compute hidden layer activations
    # Tanh is bounded and smooth, avoiding numerical overflow issues common with exp
    # We add a small epsilon to slopes to avoid exactly zero scaling if needed, 
    # but tanh(0*x+b) is just a constant, which is fine.
    
    z1 = v1 * x + b1
    z2 = v2 * x + b2
    z3 = v3 * x + b3
    
    # Clip z values to prevent tanh from saturating completely and causing 
    # vanishing gradients if the optimizer pushes values too far, 
    # though tanh is generally stable. 
    # However, if v is very large, z can be huge. tanh handles it, but 
    # gradients vanish. Clipping helps BFGS navigate.
    # A range of [-20, 20] is safe for tanh (tanh(20) ~ 1.0).
    z1 = np.clip(z1, -20, 20)
    z2 = np.clip(z2, -20, 20)
    z3 = np.clip(z3, -20, 20)
    
    a1 = np.tanh(z1)
    a2 = np.tanh(z2)
    a3 = np.tanh(z3)
    
    # Combine
    y = w1 * a1 + w2 * a2 + w3 * a3 + c
    
    # Ensure finite output
    return np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
