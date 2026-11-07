import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT = 145.0

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    if time.time() - _START_TIME > _TIMEOUT:
        return np.zeros_like(x)
    
    x = np.asarray(x).flatten()
    
    # 10 RBF centers evenly spaced in [-3, 3]
    # Using np.linspace to ensure coverage
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Varying widths (sigmas) to capture different scales of features
    # Smaller sigmas for high frequency, larger for low frequency
    # We can use a fixed set of sigmas or derive them.
    # Let's use a mix: alternating or scaled.
    # To keep it simple and robust, let's use sigmas that vary slightly
    # or are fixed to cover the range well.
    # A common choice is to have sigmas proportional to the distance to neighbors
    # or just a fixed set. Let's use a fixed set of sigmas that decrease
    # from center to edges or vice versa, or just uniform with some variation.
    # For maximum flexibility, we could make widths parameters, but we only have 10 params total.
    # So widths must be fixed.
    
    # Let's use sigmas that are roughly half the distance to the nearest neighbor
    # or just a fixed array of sigmas.
    # Distance between centers is 6/9 = 0.666.
    # Let's use sigmas around 0.5 to 1.0.
    sigmas = np.ones(10) * 0.8
    
    # Compute RBF terms: exp(-((x - c_i)^2) / (2 * sigma_i^2))
    # x shape: (N,)
    # centers shape: (10,)
    # diff shape: (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # sigma_i needs to be broadcastable
    # sigmas shape: (10,) -> (1, 10)
    sigmas_sq = sigmas[np.newaxis, :] ** 2
    
    rbf_terms = np.exp(-diff**2 / (2.0 * sigmas_sq))
    
    return rbf_terms @ params
