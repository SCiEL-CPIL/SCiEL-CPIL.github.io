"""Rewritten acquisition function with cost-aware EI and adaptive exploration."""
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0  # seconds


def _remaining_time():
    return _DEADLINE - (time.time() - _START_TIME)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with adaptive exploration for Bayesian optimization of cosine8 8d.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviations
    f_best : float - best observed value so far
    cost : array of shape (2000,) - evaluation costs
    
    Returns:
    --------
    array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard: if we're running out of time, return a simple fast score
    if _remaining_time() < 5.0:
        # Fast fallback: just EI/cost
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -15, 15)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        cost_safe = np.maximum(cost, 1e-8)
        scores = ei / cost_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -15, 15)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware: normalize by cost
    cost_safe = np.maximum(cost, 1e-8)
    ei_cost = ei / cost_safe
    
    # Adaptive exploration component:
    # Use a GP-based exploration score that rewards uncertainty
    # but is tempered by how close we are to the known best
    # The idea: explore more when we have little information (high sigma)
    # and when the mean is not already very good
    
    # Exploration score: sigma / cost, but scaled by how far mu is from f_best
    # Points with high sigma and moderate mu (not already great) get explored
    # We use a soft threshold: points where mu is within ~1 sigma of f_best
    # are "interesting" for exploration
    
    # Relative improvement potential
    # If mu is already below f_best by a lot, exploitation is better
    # If mu is near f_best, exploration might find something better
    # If mu is much worse than f_best, low priority unless sigma is very high
    
    # Compute a combined score:
    # Primary: EI/cost (exploitation)
    # Secondary: sigma/cost * exploration_weight (exploration)
    # The exploration weight adapts based on how much the GP is uncertain
    # compared to the scale of the problem
    
    # Scale of the problem: cosine8 has range roughly [-0.8, 1.0], so ~1.8
    # But standardized targets make mu and sigma in objective units
    # Use f_best as a reference for what's achievable
    
    # Adaptive beta: larger when sigma is large relative to the range of mu
    mu_range = np.max(mu) - np.min(mu)
    if mu_range < 1e-10:
        mu_range = 1e-10
    
    # Beta scales with uncertainty relative to the observed spread
    # When sigma is large relative to mu spread, we should explore more
    sigma_mean = np.mean(sigma_safe)
    beta = 0.3 * (sigma_mean / (mu_range + 1e-10))
    beta = np.clip(beta, 0.1, 1.0)
    
    # UCB-like exploration: -mu + beta * sigma (for minimization, lower is better)
    # Score = -mu + beta * sigma, normalized by cost
    ucb_score = (-mu + beta * sigma_safe) / cost_safe
    
    # Normalize both scores to comparable scales using z-scoring
    ei_mean = np.mean(ei_cost)
    ei_std = np.std(ei_cost)
    if ei_std < 1e-10:
        ei_std = 1e-10
    ei_norm = (ei_cost - ei_mean) / ei_std
    
    ucb_mean = np.mean(ucb_score)
    ucb_std = np.std(ucb_score)
    if ucb_std < 1e-10:
        ucb_std = 1e-10
    ucb_norm = (ucb_score - ucb_mean) / ucb_std
    
    # Combine: primarily EI, with adaptive exploration
    # Weight: 70% EI, 30% exploration
    # The exploration weight is higher when we have less data (early on)
    # We can approximate "less data" by the average sigma being high
    # High average sigma -> more exploration needed
    
    # Determine exploration weight based on average sigma relative to f_best scale
    # If f_best is close to the minimum (-0.8), we're doing well, less exploration
    # If f_best is far from minimum, more exploration
    # cosine8 minimum is -0.8
    min_possible = -0.8
    progress = (f_best - min_possible) / (1.0 - min_possible + 1e-10)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Early on (progress near 1), explore more; late on (progress near 0), exploit more
    exploration_weight = 0.2 + 0.3 * progress
    exploration_weight = np.clip(exploration_weight, 0.15, 0.5)
    
    combined = (1.0 - exploration_weight) * ei_norm + exploration_weight * ucb_norm
    
    # Ensure finite values
    combined = np.where(np.isfinite(combined), combined, 0.0)
    
    return combined
