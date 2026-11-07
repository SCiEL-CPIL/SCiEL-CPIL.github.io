"""Symbolic regression with 10-term RBF (Gaussian) basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(x)

    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.flatten()
    p = np.asarray(params, dtype=np.float64)
    
    if p.shape[0] != 10:
        p = p[:10] if p.shape[0] >= 10 else np.pad(p, (0, 10 - p.shape[0]))

    # 10 RBF centers evenly spaced across [-3, 3]
    # Centers: -3, -2.4, -1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8, 2.4
    # Actually, let's use linspace from -3 to 3 with 10 points for better coverage
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Fixed width for all Gaussians. 
    # A width of ~1.0 is reasonable for 200 points over a range of 6.
    # We use exp(-0.5 * ((x - c) / sigma)^2) form, so sigma is the std dev.
    # Let's try sigma = 1.0.
    sigma = 1.0
    
    # Compute basis matrix: (n, 10)
    # Each column is a Gaussian centered at centers[j]
    # phi_j(x) = exp(-0.5 * ((x - centers[j]) / sigma)^2)
    
    # Vectorized computation:
    # diff = x[:, None] - centers[None, :]  # (n, 10)
    # phi = np.exp(-0.5 * (diff / sigma)**2)
    
    diff = x[:, None] - centers[None, :]
    phi = np.exp(-0.5 * (diff / sigma)**2)
    
    # prediction = phi @ p
    pred = phi @ p
    
    return pred
# EVOLVE-BLOCK-END
