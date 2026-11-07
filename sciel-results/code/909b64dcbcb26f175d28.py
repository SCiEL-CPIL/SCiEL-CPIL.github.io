"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
import numpy as np
from scipy.stats import norm
import time

# Global timer for safety
_START_TIME = time.time()
_TIME_LIMIT = 295.0  # Safety margin below 330s wall-clock limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining EI and LCB components.
    
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
    
    # Expected Improvement (for minimization)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative (numerical issues can cause tiny negative values)
    ei = np.maximum(ei, 0.0)
    
    # Cost normalization: cost is 1 + mean(x_scaled) where x_scaled in [0,1]
    # So cost is in [1, 2]. We use EI per unit cost.
    cost_safe = np.maximum(cost, 1e-9)
    
    # Cost-aware EI component
    ei_score = ei / cost_safe
    
    # LCB component: -mu + beta * sigma
    # Use an adaptive beta that scales with the maximum uncertainty
    # This allows the algorithm to dynamically balance exploration
    max_sigma = np.max(sigma_safe)
    beta = 2.0 * max_sigma
    lcb_score = -mu + beta * sigma_safe
    
    # Combine: primarily cost-aware EI, with a smaller LCB term
    # This structural change adds exploration via LCB while maintaining cost-awareness
    weight_ei = 0.7
    weight_lcb = 0.3
    
    scores = weight_ei * ei_score + weight_lcb * lcb_score / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
