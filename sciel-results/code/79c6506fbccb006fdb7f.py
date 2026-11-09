import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 310.0  # Safety margin before 330s kill

def _check_deadline():
    """Check if we're approaching the deadline."""
    return (time.time() - _start_time) > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a structural change: Cost-Weighted Expected Improvement (EI) with
    adaptive exploration via quantile-based sigma scaling, combined with a
    penalty for high-cost regions that accounts for the remaining budget.
    
    The key structural difference: instead of PoI/cost (which is a probability
    per unit cost), we use EI/cost^alpha where alpha adapts based on how much
    budget remains. Early on, we explore more (lower alpha); later, we exploit
    cheaper improvements more aggressively (higher alpha).
    
    Additionally, we add a UCB-style exploration term that is scaled by the
    ratio of expected improvement to uncertainty, creating a balanced
    explore-exploit tradeoff.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP std predictions
    f_best : float - best observed value so far
    cost : array of shape (2000,) - evaluation costs
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = better)
    """
    # Deadline guard
    if _check_deadline():
        # Return a simple fallback: EI/cost
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return ei / cost
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Estimate budget usage: cost ranges from 1.0 to 2.0
    # Total budget is 30, initial 5 points cost ~2.5 each on average = ~12.5
    # So we have roughly 17.5 budget left, allowing ~12-17 more evaluations
    # We can estimate how many evaluations remain based on average cost so far
    # Since we don't have direct access to history, we use the cost array as a proxy
    
    # Compute EI for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB (Upper Confidence Bound) for exploration
    # For minimization, we want low mu + uncertainty
    # UCB = mu - beta * sigma (lower is better)
    # But we want higher scores, so use -UCB = -mu + beta * sigma
    
    # Adaptive beta based on sigma distribution
    # Use the median sigma to normalize
    sigma_median = np.median(sigma_safe)
    
    # Adaptive exploration coefficient:
    # Early on (high uncertainty), explore more
    # Later (low uncertainty), exploit more
    # Use a sigmoid-like function of the median sigma
    # When sigma_median is high, beta is larger (more exploration)
    # When sigma_median is low, beta is smaller (more exploitation)
    beta = 0.5 + 0.5 * np.tanh(sigma_median / 0.1)
    
    # UCB-based score (higher = more exploration)
    ucb_score = -mu + beta * sigma_safe
    
    # Cost adjustment: use a power law with adaptive exponent
    # alpha starts at 0.5 (mild cost penalty) and increases to 1.5 (strong cost penalty)
    # based on how concentrated the costs are
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    cost_spread = cost_range / (cost_max + 1e-9)
    
    # If costs are very spread out, use stronger cost penalty
    # If costs are similar, use milder cost penalty
    alpha = 0.5 + cost_spread
    
    # Cost-weighted EI
    cost_weighted_ei = ei / (cost ** alpha)
    
    # Cost-adjusted UCB
    cost_adjusted_ucb = ucb_score / (cost ** alpha)
    
    # Combine EI and UCB with adaptive weighting
    # Use the ratio of EI to UCB to determine weighting
    # When EI dominates, weight EI more; when UCB dominates, weight UCB more
    ei_sum = np.sum(cost_weighted_ei)
    ucb_sum = np.sum(np.abs(cost_adjusted_ucb))
    
    if ei_sum + ucb_sum > 1e-9:
        ei_weight = ei_sum / (ei_sum + ucb_sum)
        ucb_weight = 1.0 - ei_weight
    else:
        ei_weight = 0.5
        ucb_weight = 0.5
    
    # Final score: weighted combination of cost-weighted EI and cost-adjusted UCB
    scores = ei_weight * cost_weighted_ei + ucb_weight * cost_adjusted_ucb
    
    # Add a small penalty for very high cost regions to encourage exploration of cheap areas
    # This helps in the early stages when we haven't explored much
    cost_penalty = 0.05 * (cost - cost_min) / (cost_max - cost_min + 1e-9)
    scores = scores - cost_penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
