"""Cost-aware acquisition function for Bayesian optimization on rastrigin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_DEADLINE = 280.0
_start_time = time.time()


def _check_deadline():
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Quantile of the Posterior (QoP) acquisition for minimizing rastrigin 2d.
    
    Uses the p-th quantile of the posterior: q = mu + sigma * Phi^{-1}(p).
    Score = -q / cost^gamma, with a small UCB exploration term to avoid
    getting trapped in local minima of the quantile surface.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled)).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    cost_safe = np.maximum(cost, 1e-6)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Fixed quantile p=0.25 (cell finding: better than adaptive)
    p = 0.25
    z_p = norm.ppf(p)  # negative, ~-0.6745
    
    # Quantile of the posterior
    quantile = mu + sigma_safe * z_p
    
    # Cost exponent: stronger penalty for expensive evaluations
    gamma = 1.5
    cost_penalty = cost_safe ** gamma
    
    # Main score: minimize quantile, penalize by cost
    scores = -quantile / cost_penalty
    
    # Small exploration bonus: UCB term to encourage visiting uncertain regions
    # This prevents the quantile surface from having local minima that trap search
    # The coefficient is small so exploitation dominates
    sigma_range = sigma_safe.max() - sigma_safe.min()
    if sigma_range > 1e-12:
        # Normalize sigma to [0, 1] for the exploration term
        sigma_norm = (sigma_safe - sigma_safe.min()) / sigma_range
        # Small exploration weight: 0.05 of the typical score magnitude
        typical_score = np.abs(scores).max()
        if typical_score > 1e-12:
            scores += 0.05 * typical_score * sigma_norm
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
