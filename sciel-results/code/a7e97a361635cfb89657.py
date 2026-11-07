"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 330.0  # seconds
_SAFETY_MARGIN = 5.0  # seconds


def _is_deadline_exceeded():
    return (time.time() - _start_time) > (_DEADLINE - _SAFETY_MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Expected Improvement (EI) with UCB exploration.
    
    Computes EI = E[max(f_best - f(x), 0)] / cost, plus a UCB term for exploration.
    EI directly measures expected improvement over the best observed value,
    making it a principled choice for minimization. Normalizing by cost makes
    it cost-aware. The UCB term adds controlled exploration based on uncertainty.
    """
    # Deadline guard: if we're close to the limit, return a simple fallback
    if _is_deadline_exceeded():
        # Return scores that prioritize low cost and low predicted mean
        return -(mu + 0.1 * cost)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Cost normalization
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score for EI
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    # This is the expected amount by which f(x) improves over f_best
    from scipy.stats import norm as _norm
    EI = (f_best - mu) * _norm.cdf(z) + sigma_safe * _norm.pdf(z)
    
    # Cost-aware EI: divide by cost to get improvement per unit cost
    cost_aware_ei = EI / cost_safe
    
    # UCB exploration term: beta * sigma / cost
    # This encourages exploring uncertain regions, weighted by cost
    # Use a moderate beta to balance exploration
    beta = 0.5
    ucb_term = beta * sigma_safe / cost_safe
    
    # Final score: cost-aware EI + UCB exploration
    # Higher scores are preferred (we maximize this)
    final_score = cost_aware_ei + ucb_term
    
    # Ensure finite values
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    return final_score
# EVOLVE-BLOCK-END
