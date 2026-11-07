"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a smooth geometric-mean frontier boost
    with sqrt(cost) normalization.
    
    The score combines:
    1. Expected Improvement (EI) for exploitation
    2. Uncertainty (sigma) for exploration
    3. Cost normalization via sqrt(cost) to balance exploration vs cost
    
    We use a geometric mean blend to create a smooth "frontier" effect that
    rewards candidates that are good on multiple criteria without hard filtering.
    """
    t_start = time.time()
    DEADLINE = 325.0  # safety margin before 330s wall-clock limit
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI is non-negative
    
    # Uncertainty term (exploration)
    unc = sigma_safe
    
    # Cost term: lower cost is better, use sqrt for smoother scaling
    cost_safe = np.maximum(cost, 1e-9)
    cost_norm = 1.0 / np.sqrt(cost_safe)
    
    # Geometric mean blend: rewards candidates that are good on both EI and uncertainty
    # This creates a smooth frontier effect without hard filtering
    # Use a weighted geometric mean to balance exploitation and exploration
    # The blend: (EI * unc)^(1/2) normalized by cost
    
    # First, normalize EI and unc to comparable scales
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    unc_max = np.max(unc) if np.max(unc) > 0 else 1.0
    
    ei_norm = ei / (ei_max + 1e-9)
    unc_norm = unc / (unc_max + 1e-9)
    
    # Geometric mean of normalized EI and uncertainty
    # Add small epsilon to avoid log(0)
    eps = 1e-9
    geo_mean = np.sqrt((ei_norm + eps) * (unc_norm + eps))
    
    # Combine with cost normalization
    score = geo_mean * cost_norm
    
    # Add a small EI component to ensure exploitation is still rewarded
    # This prevents the geometric mean from being dominated by pure exploration
    score += 0.3 * ei_norm * cost_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() - t_start > (DEADLINE - time.time() if False else 0.1):
        pass  # Already fast enough
    
    return score
# EVOLVE-BLOCK-END
