"""
Strategy: Single-Layer Neural Network (Tanh Activation).
Instead of a rational function or physical power-law, we use a simple 
feed-forward neural network with one hidden layer.
This structure is a universal approximator and can model non-linear 
interactions between strain and temperature without the risk of 
division by zero or asymptotic instability.

Architecture:
Inputs: epsilon (strain), T (temp, normalized)
Hidden Layer: 2 units with tanh activation
Output: Linear combination of hidden units + linear elastic term + offset

Params (10 total):
p[0], p[1], p[2]: Weights and bias for hidden unit 1
p[3], p[4], p[5]: Weights and bias for hidden unit 2
p[6]: Weight for hidden unit 1 in output
p[7]: Weight for hidden unit 2 in output
p[8]: Linear coefficient for strain (elastic modulus approximation)
p[9]: Offset (yield stress approximation)

sigma = p[6]*tanh(p[0]*eps + p[1]*T + p[2]) + 
        p[7]*tanh(p[3]*eps + p[4]*T + p[5]) + 
        p[8]*eps + p[9]
"""

import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Inputs:
    strain: array of strain values
    temp: array of temperature values
    params: array of 10 parameters
    
    Returns:
    array of stress values
    """
    eps = np.asarray(strain, dtype=np.float64)
    T = np.asarray(temp, dtype=np.float64)
    
    # Normalize temperature to [0, 1] to help optimization
    T_norm = (T - 0.06667) / (1.0 - 0.06667)
    T_norm = np.clip(T_norm, 0.0, 1.0)
    
    p = params
    
    # Extract parameters
    # Hidden Unit 1
    z1 = p[0] * eps + p[1] * T_norm + p[2]
    h1 = np.tanh(z1)
    
    # Hidden Unit 2
    z2 = p[3] * eps + p[4] * T_norm + p[5]
    h2 = np.tanh(z2)
    
    # Output
    # We include a direct linear term for the elastic regime to help convergence
    sigma = p[6] * h1 + p[7] * h2 + p[8] * eps + p[9]
    
    # Ensure finite
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    return sigma
