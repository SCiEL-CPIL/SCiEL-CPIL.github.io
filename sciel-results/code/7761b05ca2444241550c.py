"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: rank-normalized EI + UCB with adaptive exploration.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,) - posterior mean in objective units
    sigma : ndarray, shape (2000,) - posterior std in objective units
    f_best : float - lowest observed value so far
    
    Returns: array of shape (2000,) - scores, higher means evaluate next
    """
    start_time = time.time()
    DEADLINE = start_time + 295.0  # Safety margin before 330s limit
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = np.maximum(sigma, 1e-12)  # Avoid division by zero
    
    # Clip extreme values for numerical stability
    mu = np.clip(mu, f_best - 100, f_best + 100)
    sigma = np.clip(sigma, 1e-10, 100)
    
    # Compute z-scores
    z = (f_best - mu) / sigma
    z = np.clip(z, -20, 20)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # UCB component: for minimization, we want low mu + high sigma
    beta = 2.5
    ucb = mu + beta * sigma  # This is what we want to MINIMIZE
    ucb_score = -ucb  # Negate so higher is better
    
    ucb_score = np.where(np.isfinite(ucb_score), ucb_score, -np.inf)
    
    # Also compute a pure exploration score based on sigma alone
    # This helps in early stages when we haven't seen much yet
    exploration = sigma
    exploration = np.where(np.isfinite(exploration), exploration, 0.0)
    
    # Rank-based normalization for robustness
    n = len(ei)
    
    def rank_normalize(arr):
        """Rank-based normalization to [0, 1]."""
        if len(arr) <= 1:
            return np.ones_like(arr)
        # Handle -inf values by assigning them rank 0
        mask = np.isfinite(arr)
        if not mask.any():
            return np.zeros_like(arr)
        if mask.sum() < len(arr):
            # Sort only finite values, assign rank 0 to -inf
            finite_vals = arr[mask]
            ranks_finite = np.argsort(np.argsort(finite_vals)) / (len(finite_vals) - 1)
            result = np.zeros_like(arr)
            result[mask] = ranks_finite
            return result
        ranks = np.argsort(np.argsort(arr)) / (n - 1)
        return ranks
    
    ei_ranks = rank_normalize(ei)
    ucb_ranks = rank_normalize(ucb_score)
    expl_ranks = rank_normalize(exploration)
    
    # Adaptive weighting: more exploration early, more exploitation later
    # Use the spread of mu as a proxy for how much we've explored
    mu_spread = np.std(mu)
    f_range = max(abs(f_best), 1.0)
    exploration_frac = np.clip(mu_spread / (f_range + mu_spread), 0.2, 0.6)
    
    w_ei = 1.0 - exploration_frac
    w_ucb = exploration_frac * 0.7
    w_expl = exploration_frac * 0.3
    
    # Combined score
    score = w_ei * ei_ranks + w_ucb * ucb_ranks + w_expl * expl_ranks
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() > DEADLINE:
        # Fallback: return pure EI if we're running out of time
        return np.where(np.isfinite(ei), ei, 0.0)
    
    return score
# EVOLVE-BLOCK-END
