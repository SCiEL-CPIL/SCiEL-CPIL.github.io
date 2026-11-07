"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) with uncertainty damping.
    
    Parameters:
    mu: (N,) array of posterior means
    sigma: (N,) array of posterior standard deviations
    f_best: float, the best objective value found so far
    
    Returns:
    scores: (N,) array of acquisition scores (higher is better)
    """
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 280.0  # 50s safety margin
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Handle zero or near-zero sigma to avoid division by zero in Z calculation
    # If sigma is very small, the term is dominated by (f_best - mu)
    # We use a small epsilon to stabilize division
    eps = 1e-8
    sigma_safe = np.where(sigma > eps, sigma, eps)
    
    # Calculate Z-score for Expected Improvement
    # Z = (f_best - mu) / sigma
    # Note: In minimization, improvement is f_best - Y.
    # If Y < f_best, improvement is positive.
    # E[max(0, f_best - Y)] where Y ~ N(mu, sigma)
    # = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    Z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # Phi(Z) is norm.cdf(Z)
    # phi(Z) is norm.pdf(Z)
    ei = (f_best - mu) * norm.cdf(Z) + sigma_safe * norm.pdf(Z)
    
    # For points where sigma was effectively zero (clipped to eps), 
    # the EI formula might behave oddly if mu > f_best (negative improvement).
    # EI should be 0 if mu >= f_best and sigma is tiny.
    # The formula above handles this: if mu > f_best, Z is negative large, cdf->0, pdf->0.
    # If mu < f_best, Z is positive large, cdf->1, pdf->0, ei -> f_best - mu.
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=0.0)
    
    # Structural Refinement: Uncertainty Damping
    # Dampen scores for points with extremely high uncertainty relative to the batch.
    # This prevents the algorithm from picking points that are just "noisy" 
    # rather than promising.
    sigma_max = np.max(sigma) if len(sigma) > 0 else 1.0
    
    if sigma_max > 1e-10:
        sigma_norm = sigma / sigma_max
        # Identify points in the top 5% of uncertainty
        # We want to dampen these slightly to prefer moderate uncertainty regions
        # where the GP is uncertain but the mean might be low.
        threshold = 0.95
        damp_mask = sigma_norm > threshold
        
        if np.any(damp_mask):
            # Calculate excess uncertainty
            excess = (sigma_norm[damp_mask] - threshold) / (1.0 - threshold)
            # Apply a multiplicative penalty (dampening)
            # The more extreme the uncertainty, the more we dampen the EI score.
            # Factor: 1.0 - 0.5 * excess  (reduces score by up to 50% for max sigma)
            damp_factor = 1.0 - 0.5 * excess
            ei[damp_mask] *= damp_factor
    
    # Ensure finite scores
    scores = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
