import numpy as np
from scipy.stats import norm
import time

# Global timer for safety
_START_TIME = time.time()
_TIME_LIMIT = 300.0  # Safety margin below 330s

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Cost-Aware EI and Cost-Aware UCB.
    Returns scores of shape (2000,), where higher scores indicate better candidates.
    """
    # Explicit deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: Simple Cost-Aware EI
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        cost_safe = np.maximum(cost, 1e-9)
        return ei / cost_safe

    # Ensure safety in denominators
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # 1. Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-Aware EI
    score_ei = ei / cost_safe

    # 2. Cost-Aware UCB (Upper Confidence Bound for minimization)
    # We want to maximize the lower bound of the potential minimum.
    # Lower bound of f(x) is mu(x) - beta * sigma(x).
    # We want to minimize f, so we maximize -(mu - beta*sigma) = beta*sigma - mu.
    # However, standard UCB for minimization often uses:
    # score = -mu + beta * sigma
    # Let's use a beta that balances exploration.
    beta = 2.0
    ucb = -mu + beta * sigma_safe
    
    # Cost-Aware UCB
    score_ucb = ucb / cost_safe

    # 3. Hybrid Score
    # Combine EI and UCB. EI is better for exploitation, UCB for exploration.
    # A weighted sum is smoother than max.
    # Weight 1.0 for EI and 0.5 for UCB (scaled to be comparable)
    # Note: EI and UCB have different scales. EI is in objective units. UCB is in objective units.
    # Let's try a simple weighted sum.
    scores = score_ei + 0.5 * score_ucb
    
    # Ensure all outputs are finite
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores
