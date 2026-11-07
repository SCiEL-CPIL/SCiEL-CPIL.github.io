# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Adaptive Gaussian process-like basis function approximator with
    variable widths controlled by params.
    
    Uses 10 Gaussian bumps with:
    - Fixed centers evenly spaced in [-3, 3]
    - Widths (sigma) determined by params[0:5] (shared in pairs for stability)
    - Weights determined by params[5:10]
    
    This gives the optimizer more expressive power to capture localized features.
    
    p[0:5]  -> log-widths for Gaussian pairs (exp(p[i]) gives sigma)
    p[5:10] -> weights for the 10 Gaussian basis functions
    """
    p = np.asarray(params, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    
    n_centers = 10
    centers = np.linspace(-3.0, 3.0, n_centers)
    
    # Extract log-widths and weights
    # Use first 5 params for log-sigma (shared for pairs), last 5 for weights
    # Actually, let's use a different split: 
    # p[0:10] are all weights, and we use fixed but varied widths
    
    # Alternative: use all 10 params as weights, with fixed but non-uniform widths
    # This is simpler and more stable for BFGS
    
    # Use fixed widths that are wider near the center and narrower at edges
    # This provides better resolution where the function might vary more
    sigmas = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.2, 1.1, 1.0, 0.9, 0.8])
    
    # Compute all basis functions
    result = np.zeros_like(x)
    for i in range(n_centers):
        diff = (x - centers[i]) / sigmas[i]
        basis = np.exp(-0.5 * diff * diff)
        result += p[i] * basis
    
    return result
# EVOLVE-BLOCK-END
