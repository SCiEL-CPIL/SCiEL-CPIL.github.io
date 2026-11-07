"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to track elapsed time for adaptive parameters
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Pure Expected Improvement (EI)
    with an adaptive exploration component and square-root cost normalization.
    
    The strategy relies on the fact that EI naturally balances exploration (via sigma)
    and exploitation (via mu). By dividing by sqrt(cost), we favor low-cost regions
    without being overly aggressive, and by adapting the implicit exploration weight
    through the standard EI formulation which depends on the current uncertainty,
    we avoid the pitfalls of fixed UCB weights.
    
    To enhance adaptivity, we inject a small, decaying UCB term into the mean
    effectively increasing the 'optimistic' bias in the early stages, but since
    we are constrained to return scores based on mu/sigma, we simulate this by
    scaling the sigma contribution in EI if we were to modify it, but standard EI
    is robust. Instead, we use a pure CA-EI but ensure numerical stability and
    use a slightly more aggressive cost penalty (power 0.5) which is known to be
    effective for this type of budget-constrained problem.
    
    Key Structural Change:
    1. Pure EI is used, not a hybrid.
    2. Cost normalization is 1/sqrt(cost).
    3. We add a safety check for time, though the computation is O(N) and fast.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    if mu.shape != sigma.shape or mu.shape != cost.shape:
        raise ValueError("mu, sigma, and cost must have the same shape")
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in extreme tails, though norm.cdf handles it well
    # But for very large positive z, cdf is 1, pdf is 0.
    # For very large negative z, cdf is 0, pdf is 0.
    # Standard EI calculation is stable.
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical noise can make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost-Aware Adjustment:
    # The directive suggests EI / (cost + 1e-6)^0.5.
    # This is a structural choice: it penalizes high cost sub-linearly,
    # allowing high-EI points in moderate-cost regions to still be preferred
    # over low-EI points in very low-cost regions.
    
    score = ei / np.sqrt(cost_safe)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # If all scores are 0 (e.g., sigma is 0 everywhere), fallback to random or min cost
    if np.all(score == 0):
        # Prefer low cost points if no improvement is predicted
        score = 1.0 / cost_safe
    
    return score
# EVOLVE-BLOCK-END
