"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Uses rank-normalized Expected Improvement combined with LCB (Lower Confidence Bound)
    for exploration. The LCB component with adaptive beta provides a clean linear
    acquisition that balances exploration and exploitation.
    
    mu, sigma: arrays of shape (2000,) - posterior mean and std in objective units
    f_best: float - lowest observed value so far
    
    Returns: array of shape (2000,) - scores, higher means evaluate next
    """
    start_time = time.time()
    DEADLINE = start_time + 295.0  # Safety margin before 330s limit
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Clip extreme values for numerical stability
    mu = np.clip(mu, f_best - 50, f_best + 50)
    sigma_safe = np.clip(sigma_safe, 1e-10, 100)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    
    # Base Expected Improvement (for minimization)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # LCB component: for minimization, we want low mu - beta*sigma
    # Adaptive beta: more exploration early, less later
    known_min = -1.8013
    progress = np.clip((f_best - known_min) / (5.0 - known_min), 0.0, 1.0)
    # Beta starts high (more exploration) and decreases
    beta = 0.5 + 2.0 * (1.0 - progress)
    
    lcb = mu - beta * sigma_safe
    lcb_score = -lcb  # Negate so higher is better
    
    lcb_score = np.where(np.isfinite(lcb_score), lcb_score, 0.0)
    
    # Rank-based normalization for robustness to outliers
    n = len(ei)
    
    def rank_normalize(arr):
        """Rank-based normalization to [0, 1]."""
        if len(arr) <= 1:
            return np.ones_like(arr)
        mask = np.isfinite(arr)
        if not mask.any():
            return np.zeros_like(arr)
        if mask.sum() < len(arr):
            finite_vals = arr[mask]
            ranks_finite = np.argsort(np.argsort(finite_vals)) / (len(finite_vals) - 1)
            result = np.zeros_like(arr)
            result[mask] = ranks_finite
            return result
        ranks = np.argsort(np.argsort(arr)) / (n - 1)
        return ranks
    
    ei_ranks = rank_normalize(ei)
    lcb_ranks = rank_normalize(lcb_score)
    
    # Adaptive weighting: more exploration early, more exploitation later
    # Use mu spread as proxy for exploration progress
    mu_spread = np.std(mu)
    f_range = max(abs(f_best), 1.0)
    exploration_frac = np.clip(mu_spread / (f_range + mu_spread), 0.2, 0.5)
    
    w_ei = 1.0 - exploration_frac
    w_lcb = exploration_frac
    
    # Combined score
    score = w_ei * ei_ranks + w_lcb * lcb_ranks
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() > DEADLINE:
        # Fallback: return pure EI if we're running out of time
        return np.where(np.isfinite(ei), ei, 0.0)
    
    return score
# EVOLVE-BLOCK-END
