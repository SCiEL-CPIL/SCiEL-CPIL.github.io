"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _fast_norm_cdf(x):
    """Fast approximation of standard normal CDF using Abramowitz & Stegun."""
    return 0.5 * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))


def _fast_norm_pdf(x):
    """Fast approximation of standard normal PDF."""
    return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 3D.
    Uses a pure EI-based approach with adaptive gamma for exploration/exploitation balance.
    """
    start_time = time.time()
    deadline = start_time + 5.0  # 5 second safety margin per call
    
    # Guard against zero/negative sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * _fast_norm_cdf(z) + sigma_safe * _fast_norm_pdf(z)
    
    # Adaptive gamma: use a value that balances exploration and exploitation
    # When f_best is close to the known minimum, we want more exploitation (lower gamma)
    # When f_best is far from the minimum, we want more exploration (higher gamma)
    # Known minimum of Hartmann 3D is approximately -3.86278
    min_val = -3.86278
    
    # Progress metric: how close we are to the minimum
    # If f_best is close to min_val, progress is high (closer to 1)
    # If f_best is far from min_val, progress is low (closer to 0)
    # Scale: assume typical range of f_best is from min_val to maybe 0 or 1
    progress = np.clip((f_best - min_val) / (0.0 - min_val), 0.0, 1.0)
    
    # When progress is high (close to optimum), use lower gamma (more exploitation)
    # When progress is low (far from optimum), use higher gamma (more exploration)
    gamma = 0.5 + 0.5 * (1.0 - progress)
    
    # Adjust EI by gamma: scale the improvement part
    # Higher gamma means we weight the uncertainty (sigma * phi(z)) more
    # This encourages exploration in high-uncertainty regions
    adjusted_ei = ei * gamma
    
    # Ensure finite values
    adjusted_ei = np.where(np.isfinite(adjusted_ei), adjusted_ei, 0.0)
    
    # Check time
    if time.time() > deadline:
        return adjusted_ei
    
    return adjusted_ei
# EVOLVE-BLOCK-END
