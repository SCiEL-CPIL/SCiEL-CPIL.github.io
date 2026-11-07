"""
Symbolic regression with fixed non-linear feature extraction.
Uses a fixed random neural network to generate 9 features, plus a bias term.
The 10 parameters are fitted by the evaluator to linearly combine these features.
"""
import numpy as np

# Fixed random weights for the feature extractor
# We use a fixed seed to ensure reproducibility across evaluations
_rng = np.random.RandomState(12345)

# Architecture:
# Input: 2 (x, y)
# Hidden Layer 1: 16 units, tanh activation
# Hidden Layer 2: 9 units, tanh activation
# Total features: 1 (bias) + 9 (non-linear) = 10

# Hidden Layer 1: 2 -> 16
# Xavier initialization
W1 = _rng.randn(16, 2) * np.sqrt(1.0 / 2)
b1 = _rng.randn(16) * 0.1

# Hidden Layer 2: 16 -> 9
W2 = _rng.randn(9, 16) * np.sqrt(1.0 / 16)
b2 = _rng.randn(9) * 0.1

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes prediction using fixed non-linear features and linear parameters.
    """
    # Ensure inputs are 1D float64 arrays
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    
    # Stack inputs: (N, 2)
    X = np.stack([x, y], axis=-1)
    
    # Forward pass through fixed network
    # Layer 1
    Z1 = X @ W1.T + b1
    H1 = np.tanh(Z1)
    
    # Layer 2
    Z2 = H1 @ W2.T + b2
    H2 = np.tanh(Z2)
    
    # H2 shape: (N, 9)
    
    # Create feature matrix with bias
    # Bias is the first parameter p[0]
    # Non-linear features are p[1] to p[9]
    # We construct the prediction directly to avoid large matrix allocations if possible, 
    # but N=200 is small.
    
    # pred = p[0] + sum_{i=1}^{9} p[i] * H2[:, i-1]
    # H2 is (N, 9). params[1:] is (9,).
    
    # Efficient dot product
    # H2 @ params[1:] gives (N,)
    pred = params[0] + H2 @ params[1:]
    
    return pred
