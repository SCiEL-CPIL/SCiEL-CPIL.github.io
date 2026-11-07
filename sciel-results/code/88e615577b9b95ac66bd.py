"""Neural Network Model: 2-hidden-layer MLP with ReLU activations."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a small Multi-Layer Perceptron (MLP).
    
    Architecture:
    Input (3) -> Hidden Layer 1 (10 neurons, ReLU) -> Hidden Layer 2 (10 neurons, ReLU) -> Output (1)
    
    This structure provides high capacity to fit complex non-linear functions
    (e.g., products, sums of trigonometric functions, rational functions) 
    more robustly than fixed basis functions due to the universal approximation
    property and smoother gradient flow compared to RBFs with variable widths.
    
    Parameters (length 211):
    - 30 weights for Layer 1 (10x3)
    - 10 biases for Layer 1
    - 100 weights for Layer 2 (10x10)
    - 10 biases for Layer 2
    - 10 weights for Output Layer (1x10)
    - 1 bias for Output Layer
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    # Ensure params is a flat array of length 211
    params = np.asarray(params, dtype=np.float64).ravel()
    if params.size != 211:
        # Fallback to ones if size mismatch, though evaluator should provide correct size
        params = np.ones(211)

    # Reshape parameters into weights and biases
    # Layer 1: 3 inputs, 10 neurons
    W1 = params[0:30].reshape(10, 3)
    b1 = params[30:40]
    
    # Layer 2: 10 inputs, 10 neurons
    W2 = params[40:140].reshape(10, 10)
    b2 = params[140:150]
    
    # Output Layer: 10 inputs, 1 neuron
    W3 = params[150:160].reshape(10, 1)
    b3 = params[160:161]

    # Forward pass
    # Input vector: (N, 3)
    X_in = np.column_stack((x, y, z))
    
    # Hidden Layer 1
    Z1 = X_in @ W1.T + b1
    A1 = np.maximum(Z1, 0)  # ReLU
    
    # Hidden Layer 2
    Z2 = A1 @ W2.T + b2
    A2 = np.maximum(Z2, 0)  # ReLU
    
    # Output Layer
    prediction = (A2 @ W3.T + b3).ravel()
    
    # Ensure finite output
    if not np.all(np.isfinite(prediction)):
        return np.full_like(x, np.nan)
        
    return prediction
# EVOLVE-BLOCK-END
