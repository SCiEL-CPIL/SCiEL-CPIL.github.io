"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    
    # Fixed centers evenly spaced in [-2, 2]
    # 10 centers
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Fixed widths. 
    # If centers are spaced by 4/9 ~ 0.44, a width of ~0.3-0.5 is reasonable.
    # Let's use a width that provides good overlap but not too much.
    # sigma = 0.35
    sigma = 0.35
    
    # Compute RBF basis functions
    # x is (N,), centers is (10,)
    # diff is (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian basis: exp(-0.5 * (x-c)^2 / sigma^2)
    # Precompute constant
    const = 0.5 / (sigma * sigma)
    
    basis = np.exp(-const * diff**2)
    
    # params is (10,)
    # Prediction is basis @ params
    # basis is (N, 10), params is (10,)
    pred = basis @ params
    
    # Ensure finite output
    # If params are huge, exp might overflow? No, exp is bounded by 1.
    # If params are huge, pred might be huge, but that's what the optimizer decides.
    # However, if the fit fails, params might be nan/inf?
    # The evaluator handles non-finite error during fitting.
    # We just return the linear combination.
    
    return pred
# EVOLVE-BLOCK-END
