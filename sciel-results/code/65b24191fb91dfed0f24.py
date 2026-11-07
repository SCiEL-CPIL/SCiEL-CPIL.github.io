import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Two-hidden-layer tanh neural network, linear in the 10 output-layer weights.
    The internal weights are fixed (random, seeded) to keep the model linear-in-params
    for BFGS while providing universal approximation capability.
    """
    t0 = time.time()
    DEADLINE = 145.0  # 5s margin before 150s kill

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)

    # Fixed internal architecture: 2 inputs -> 20 hidden (tanh) -> 20 hidden (tanh) -> 10 outputs
    # We use the 10 params as the final output weights, and we need to produce 10 "features"
    # from the two hidden layers. Actually, let's re-think the architecture.
    
    # The strategy says: "10 parameters are the weights of the final output layer".
    # This implies the output of the second hidden layer should be of dimension 10,
    # and then we take the dot product with params.
    
    # Architecture: 
    # Input (2) -> W1 (20x2) + b1 (20) -> tanh -> H1 (20)
    # H1 (20) -> W2 (10x20) + b2 (10) -> tanh -> H2 (10)
    # H2 (10) @ params (10) -> Output (1)
    
    # We need fixed random weights for W1, b1, W2, b2.
    # To ensure reproducibility, we seed the RNG.
    
    # However, creating a new RNG every call is expensive and might not be ideal.
    # We can use a fixed seed and generate the weights once. But `equation` is called
    # repeatedly during optimization. We should cache the weights.
    
    global _nn_weights
    if not hasattr(equation, '_nn_weights'):
        rng = np.random.RandomState(42)
        # W1: (20, 2), b1: (20,)
        W1 = rng.randn(20, 2) * np.sqrt(2.0 / 2)  # He init
        b1 = np.zeros(20)
        
        # W2: (10, 20), b2: (10,)
        W2 = rng.randn(10, 20) * np.sqrt(2.0 / 20)
        b2 = np.zeros(10)
        
        equation._nn_weights = (W1, b1, W2, b2)
    
    W1, b1, W2, b2 = equation._nn_weights
    
    # Compute hidden layers
    # H1 = tanh(x @ W1.T + b1)
    # x is (N, 2), W1 is (20, 2), so x @ W1.T is (N, 20)
    H1 = np.tanh(x @ W1.T + b1)
    
    # H2 = tanh(H1 @ W2.T + b2)
    # H1 is (N, 20), W2 is (10, 20), so H1 @ W2.T is (N, 10)
    H2 = np.tanh(H1 @ W2.T + b2)
    
    # Output = H2 @ params
    # H2 is (N, 10), params is (10,)
    pred = H2 @ p
    
    # Safety check
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    # Deadline check (though this function is fast, it's good practice)
    if time.time() - t0 > DEADLINE:
        return pred
        
    return pred
