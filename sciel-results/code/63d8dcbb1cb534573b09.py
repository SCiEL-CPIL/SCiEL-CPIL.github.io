"""
Strategy: 1-Hidden-Layer Neural Network (1-2-1 architecture).
Instead of a fixed polynomial basis, we use a small neural network with 2 hidden neurons.
This allows for nonlinear feature interactions (e.g., x*y, x^2*y) that are not easily
captured by a degree-3 global polynomial with only 10 coefficients.

Architecture:
Input: x, y
Hidden Layer: 2 neurons
  h1 = tanh(w1*x + w2*y + b1)
  h2 = tanh(w3*x + w4*y + b2)
Output:
  f(x, y) = v1*h1 + v2*h2 + c

Parameters (10 total):
  w1, w2, b1: Weights and bias for neuron 1
  w3, w4, b2: Weights and bias for neuron 2
  v1, v2: Output weights
  c: Output bias
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the output of a 1-2-1 neural network.
    """
    if params is None or len(params) < 10:
        return np.zeros_like(x)
    
    # Extract parameters
    w1, w2, b1, w3, w4, b2, v1, v2, c, _ = params
    
    # Compute hidden layer activations
    # Using tanh for bounded, smooth nonlinearity
    z1 = w1 * x + w2 * y + b1
    z2 = w3 * x + w4 * y + b2
    
    # Apply activation
    h1 = np.tanh(z1)
    h2 = np.tanh(z2)
    
    # Combine outputs
    result = v1 * h1 + v2 * h2 + c
    
    return result
# EVOLVE-BLOCK-END
