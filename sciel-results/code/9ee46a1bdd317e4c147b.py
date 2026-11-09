import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Small feed-forward neural network with 10 parameters.
    Architecture:
    - Input: [x, y] (2 features)
    - Hidden Layer 1: 4 units, tanh activation
      Weights: params[0:8] (4x2 matrix, row-major: w1_1_1, w1_1_2, w1_2_1, w1_2_2, ...)
      Biases: params[8:10] is not enough for 4 biases + output layer.
      
    Wait, I have exactly 10 parameters. Let's design a minimal net that uses all 10.
    
    Option A: 1 hidden layer, 4 units.
    Weights: 4 * 2 = 8.
    Biases: 4.
    Output Weights: 4 * 1 = 4.
    Output Bias: 1.
    Total: 8 + 4 + 4 + 1 = 17 params. Too many.

    Option B: 1 hidden layer, 2 units.
    Weights: 2 * 2 = 4.
    Biases: 2.
    Output Weights: 2 * 1 = 2.
    Output Bias: 1.
    Total: 4 + 2 + 2 + 1 = 9 params.
    I have 10. I can add a second hidden layer? Or just use 9 and ignore the 10th?
    Better to use all 10 if possible, or design for exactly 10.
    
    Let's try:
    Hidden Layer 1: 2 units
    Weights (4): p[0], p[1], p[2], p[3]
    Biases (2): p[4], p[5]
    
    Hidden Layer 2: 1 unit? No, let's keep it simple.
    
    Alternative: 1 hidden layer, 3 units?
    Weights: 3 * 2 = 6.
    Biases: 3.
    Output Weights: 3.
    Output Bias: 1.
    Total: 6 + 3 + 3 + 1 = 13. Too many.

    Let's stick to 9 params and use the 10th as a scaling factor or just ignore it?
    Actually, I can make the output layer have 2 inputs from a 2-unit hidden layer,
    and add a direct skip connection?
    
    Let's do this:
    Hidden Layer: 2 units.
    Weights: 4 (p[0:4])
    Biases: 2 (p[4:6])
    Output Weights: 2 (p[6:8])
    Output Bias: 1 (p[8])
    Direct Input Weight for x? No, let's just use 9 and set p[9] to 0 or use it in a way that doesn't break things.
    
    Actually, a better use of 10 params:
    Hidden Layer 1: 2 units.
    Weights: 4 (p[0:4])
    Biases: 2 (p[4:6])
    
    Hidden Layer 2: 1 unit?
    Weights: 2 (p[6:8])
    Biases: 1 (p[8])
    
    Output: Linear?
    Weights: 1 (p[9])
    
    Total: 4 + 2 + 2 + 1 + 1 = 10.
    
    Architecture:
    h1 = tanh(X @ W1.T + b1)  # X is (N, 2), W1 is (2, 2), b1 is (2,)
    h2 = tanh(h1 @ W2.T + b2)  # h1 is (N, 2), W2 is (1, 2), b2 is (1,)
    out = h2 * p[9]            # Simple scaling
    
    Let's verify shapes:
    p[0:4] -> W1 (2x2). Reshape to (2,2).
    p[4:6] -> b1 (2,).
    p[6:8] -> W2 (1x2). Reshape to (1,2).
    p[8]   -> b2 (scalar).
    p[9]   -> scale (scalar).
    
    This allows the network to learn nonlinear features in h1, transform them in h2, and scale the result.
    """
    # Reshape parameters into network weights
    W1 = params[0:4].reshape(2, 2)
    b1 = params[4:6]
    W2 = params[6:8].reshape(1, 2)
    b2 = params[8]
    scale = params[9]
    
    # Forward pass
    # x and y are 1D arrays. Stack them into a 2D array (N, 2)
    X = np.column_stack((x, y))
    
    # Hidden Layer 1
    Z1 = X @ W1.T + b1
    H1 = np.tanh(Z1)
    
    # Hidden Layer 2
    Z2 = H1 @ W2.T + b2
    H2 = np.tanh(Z2)
    
    # Output
    pred = H2 * scale
    
    # Ensure output is 1D
    return pred.flatten()
