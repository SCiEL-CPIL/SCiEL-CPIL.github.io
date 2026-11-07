# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 2 inputs x, y in [-1, 1]^2.
    
    Strategy: Fixed-kernel neural network (single hidden layer) with 9 tanh neurons.
    This provides a non-linear, universal approximator capable of modeling complex,
    non-periodic, and asymmetric features (like sharp gradients or localized peaks)
    that linear polynomial or Fourier bases struggle to capture efficiently.
    
    Architecture:
    - Input: [x, y]
    - Hidden Layer: 9 neurons with tanh activation.
    - Output Layer: Linear combination of hidden activations + bias.
    
    Parameters (10 total):
    - params[0]: Bias term (constant).
    - params[1:10]: Weights for the 9 hidden neuron outputs.
    
    The hidden layer weights are fixed to a deterministic, well-spread configuration
    to ensure diverse feature extraction without requiring BFGS to optimize them
    (which would make the problem non-convex and harder). BFGS only fits the 10
    linear output parameters, which is a convex least-squares problem, ensuring
    fast and reliable convergence.
    """
    # Deadline guard
    t0 = time.time()
    
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Fixed hidden layer weights (9x2 matrix)
    # We choose a deterministic set of weights that provides good coverage of the input space.
    # Using a simple grid or pseudo-random fixed seed values.
    # Let's use a 3x3 grid of weights for x and y to create diverse "views" of the input.
    # w_i = [a_i, b_i] for i in 0..8
    # a_i, b_i in {-1, 0, 1}
    
    # Define 9 distinct weight vectors
    # 1. [1, 0]  -> x
    # 2. [0, 1]  -> y
    # 3. [1, 1]  -> x+y
    # 4. [1, -1] -> x-y
    # 5. [-1, 1] -> -x+y
    # 6. [-1, -1]-> -x-y
    # 7. [0.5, 0.5] -> 0.5(x+y) (scaled version for different saturation)
    # 8. [0.5, -0.5] -> 0.5(x-y)
    # 9. [1, 2]  -> x+2y (asymmetric)
    
    # To make it more general and less "hand-crafted" looking, let's use a fixed random seed
    # to generate the hidden weights, ensuring reproducibility.
    # However, hardcoding is safer for determinism across environments if RNG is not seeded consistently.
    # Let's hardcode a robust set of 9 weight vectors.
    
    # Weight matrix H (9, 2)
    # We want these to be orthogonal-ish or at least linearly independent to maximize expressiveness.
    H = np.array([
        [1.0,  0.0],
        [0.0,  1.0],
        [1.0,  1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [-1.0, -1.0],
        [0.5,  0.5],
        [0.5, -0.5],
        [2.0,  1.0]
    ])
    
    # Compute hidden layer activations
    # z = X @ H.T  where X is (N, 2), H is (9, 2) -> z is (N, 9)
    # X = column_stack([x, y])
    X = np.column_stack((x, y))
    z = X @ H.T  # Shape (N, 9)
    
    # Apply tanh activation
    h = np.tanh(z)  # Shape (N, 9)
    
    # Output layer
    # pred = h @ params[1:10] + params[0]
    
    p = params
    bias = p[0]
    weights = p[1:10]
    
    # Linear combination
    pred = h @ weights + bias
    
    # Safety check
    if not np.all(np.isfinite(pred)):
        return np.zeros_like(x)
        
    return pred
# EVOLVE-BLOCK-END
