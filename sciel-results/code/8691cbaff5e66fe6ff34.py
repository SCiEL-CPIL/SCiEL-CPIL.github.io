"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
import numpy as np
from scipy.stats import norm
import time

# Global timer for safety
_START_TIME = time.time()
_TIME_LIMIT = 295.0  # Safety margin below 330s wall-clock limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Explicit deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        # Fallback: simple cost-aware EI
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        cost_safe = np.maximum(cost, 1e-9)
        return ei / cost_safe
    
    # Convert to numpy arrays for efficiency
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-scores
    z = (f_best - mu) / sigma_safe
    
    # Clip z for numerical stability
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative (numerical issues can cause tiny negative values)
    ei = np.maximum(ei, 0.0)
    
    # Cost normalization: cost is 1 + mean(x_scaled) where x_scaled in [0,1]
    # So cost is in [1, 2].
    cost_safe = np.maximum(cost, 1e-9)
    
    # Structural change: Combine EI (exploitation) with LCB (exploration)
    # The LCB term -mu + beta*sigma provides a strong exploration signal
    # that is independent of f_best, which helps when f_best is far from the minimum.
    # The EI term provides a refined exploitation signal based on current best.
    # We blend them. The LCB score is normalized by cost as well.
    
    # Adaptive beta for LCB: scale with max uncertainty
    max_sigma = np.max(sigma_safe)
    # Use a factor that balances exploration. 1.5 is a moderate choice.
    beta = 1.5 * max_sigma
    
    # LCB score (for minimization, we maximize -mu + beta*sigma)
    lcb_score = -mu + beta * sigma_safe
    
    # Normalize LCB by cost
    lcb_cost = lcb_score / cost_safe
    
    # Normalize EI by cost
    ei_cost = ei / cost_safe
    
    # Blend: 0.5 * EI + 0.5 * LCB
    # This creates a hybrid acquisition that uses both the expected improvement
    # and the lower confidence bound, each weighted by cost.
    scores = 0.5 * ei_cost + 0.5 * lcb_cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
