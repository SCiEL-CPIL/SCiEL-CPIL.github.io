"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure x is a 1D array for safe operations
    x = np.asarray(x, dtype=np.float64)
    
    # Scale x to [-1, 1] for better numerical conditioning
    t = x / 3.0
    
    # Use RBF (Radial Basis Function) basis for local flexibility
    # This allows fitting arbitrary smooth functions better than global basis
    # Each parameter controls local behavior around its center
    
    # Fixed centers spread across the domain
    centers = np.linspace(-1, 1, 10)
    
    # Width parameter for RBFs (controls locality)
    width = 0.5
    
    # Compute RBF features: exp(-width^2 * (t - center)^2)
    # This creates a set of local bumps that can combine to form smooth functions
    features = np.zeros((len(t), 10))
    for i in range(10):
        features[:, i] = np.exp(-width**2 * (t - centers[i])**2)
    
    # Linear combination
    return features.dot(params)

# EVOLVE-BLOCK-END
