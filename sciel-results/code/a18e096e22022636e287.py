"""Symbolic regression with 1 input; Hybrid MLP + RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model combining a small MLP (1-3-1) with RBF basis functions.
    
    Parameters (length 10):
    - params[0:3]: Weights from input to hidden layer (w1)
    - params[3:6]: Biases for hidden layer (b1)
    - params[6:9]: Weights from hidden to output (w2)
    - params[9]:   Output bias (b2)
    
    The MLP captures global trends while RBF terms capture local variations.
    Input x is scaled to roughly [-1, 1] for numerical stability.
    RBF centers are fixed at 1.5, 3.0, 4.5 with sigma=1.2.
    
    Wait, 1-3-1 MLP uses 3+3+3+1 = 10 params. That leaves no params for RBF.
    
    Let's reconsider: 
    Option A: 1-2-1 MLP (2+2+2+1=7 params) + 3 RBF coefficients = 10 params
    Option B: 1-3-1 MLP (10 params) - no RBF
    Option C: Linear basis (10 params) - no nonlinearity except basis functions
    
    The inspiration with score 5.1947 used a 1-3-1 MLP with 10 params and got a good score.
    The current program uses a linear basis and got 9882 (bad).
    The inspiration with score 835 used 1-2-1 MLP + 3 RBF.
    
    Let's try a hybrid: 1-2-1 MLP (7 params) + 3 RBF terms (3 params) = 10 params.
    This gives nonlinearity from tanh AND local adaptivity from RBF.
    
    Actually, let's look at the best performing inspiration: score 5.1947 was 1-3-1 MLP.
    Score 835 was 1-2-1 MLP + 3 RBF.
    
    A 1-3-1 MLP is more expressive than 1-2-1 MLP + 3 RBF in terms of global fitting.
    But RBF terms can help with local features.
    
    Let's try: 1-2-2-1 MLP?
    Layer 1: 2 units -> 2 weights + 2 biases = 4
    Layer 2: 2 units -> 2*2 weights + 2 biases = 6
    Output: 1 unit -> 2 weights + 1 bias = 3
    Total: 4 + 6 + 3 = 13. Too many.
    
    Let's stick with 1-2-1 MLP (7 params) + 3 RBF coefficients (3 params).
    This is a clean structural change from the linear basis.
    """
    x = np.asarray(x, dtype=float).flatten()
    params = np.asarray(params, dtype=float).flatten()
    
    # Ensure exactly 10 parameters
    if len(params) < 10:
        params = np.concatenate([params, np.ones(10 - len(params))])
    else:
        params = params[:10]
    
    # Scale input x from [0.1, 6] to roughly [-1, 1]
    x_scaled = (x - 3.05) / 3.0
    
    # --- MLP Part (1-2-1) ---
    # params[0:2]: w1 (weights input to hidden)
    # params[2:4]: b1 (biases hidden)
    # params[4:6]: w2 (weights hidden to output)
    # params[6]:   b2 (output bias)
    
    w1_0, w1_1 = params[0], params[1]
    b1_0, b1_1 = params[2], params[3]
    w2_0, w2_1 = params[4], params[5]
    b2 = params[6]
    
    h1 = np.tanh(x_scaled * w1_0 + b1_0)
    h2 = np.tanh(x_scaled * w1_1 + b1_1)
    
    y_mlp = w2_0 * h1 + w2_1 * h2 + b2
    
    # --- RBF Part ---
    # params[7:10]: coefficients for 3 RBF centers
    centers = np.array([1.5, 3.0, 4.5])
    sigma = 1.2
    
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_basis = np.exp(-0.5 * (diff / sigma) ** 2)  # shape (N, 3)
    
    rbf_coeffs = params[7:10]
    
    y_rbf = rbf_basis @ rbf_coeffs
    
    # Combine
    y = y_mlp + y_rbf
    
    return y
# EVOLVE-BLOCK-END
