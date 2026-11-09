"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm
from scipy.spatial.distance import cdist

# Global deadline guard
_DEADLINE = None
def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 290  # 290s safety margin from 300s limit
    return _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Levy 2D.
    
    Uses a hybrid approach:
    1. Thompson Sampling for exploration (sample from posterior)
    2. Cost-aware scoring: balance improvement potential against evaluation cost
    3. Deadline guard for safety
    
    Parameters:
    - mu: (2000,) posterior mean in objective units
    - sigma: (2000,) posterior std in objective units  
    - f_best: scalar, lowest observed value
    - cost: (2000,) evaluation cost for each candidate
    
    Returns:
    - (2000,) scores, higher = evaluate next
    """
    t_start = time.time()
    deadline = _get_deadline()
    
    n = len(mu)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Thompson Sampling: draw a sample from the posterior for each candidate
    # Use a single random seed per call for reproducibility within the run
    rng = np.random.default_rng()
    samples = mu + sigma * rng.standard_normal(n)
    
    # Expected improvement per unit cost
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware: divide by cost
    ei_per_cost = ei / (cost + 1e-9)
    
    # Thompson sample score: how good is the sampled value relative to f_best, per cost
    ts_score = (f_best - samples) / (cost + 1e-9)
    
    # Combine: use a mixture of EI/cost and Thompson score
    # Weight towards Thompson sampling for exploration, EI/cost for exploitation
    # The mixture weight can be adaptive based on the spread of sigma
    sigma_mean = np.mean(sigma)
    sigma_max = np.max(sigma)
    # If uncertainty is high, trust Thompson more; if low, trust EI more
    exploration_weight = min(1.0, sigma_mean / (sigma_max + 1e-9) * 2.0)
    exploration_weight = 0.5 + 0.5 * exploration_weight  # between 0.5 and 1.0
    
    score = exploration_weight * ts_score + (1.0 - exploration_weight) * ei_per_cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check
    if time.time() > deadline - 5:  # 5s safety margin
        # Return a simple fallback: prefer low mu and low cost
        fallback = (f_best - mu) / (cost + 1e-9)
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    return score
# EVOLVE-BLOCK-END
