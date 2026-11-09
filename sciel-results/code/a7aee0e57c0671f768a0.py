"""Cost-aware acquisition function for Bayesian optimization on rastrigin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Set a deadline guard: we want to finish well within 300s
_DEADLINE = 280.0  # seconds
_start_time = time.time()


def _check_deadline():
    """Check if we're close to the deadline."""
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing rastrigin 2d.
    
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
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-6)
    
    # Clamp sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Structural change: Use a cost-aware variant of Expected Improvement
    # that directly incorporates the cost into the improvement calculation.
    # Instead of normalizing and combining EI and UCB separately, we compute
    # a single acquisition function that balances improvement and cost.
    
    # Compute the expected improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute a cost-aware score:
    # We want to maximize expected improvement per unit cost.
    # However, pure EI/cost can be too aggressive in exploration.
    # Instead, we use a hybrid: EI + gamma * (uncertainty term) / cost
    
    # The uncertainty term encourages exploration in areas with high sigma
    # But we scale it down to avoid excessive exploration
    gamma = 0.5
    uncertainty_term = sigma_safe
    
    # Combine EI and uncertainty term, then divide by cost
    # This gives us a cost-aware acquisition that balances exploitation and exploration
    raw_score = ei + gamma * uncertainty_term
    
    # Divide by cost to prefer cheaper evaluations
    # Use a softer cost penalty (square root) to avoid being too restrictive
    cost_penalty = np.sqrt(cost_safe)
    scores = raw_score / cost_penalty
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
