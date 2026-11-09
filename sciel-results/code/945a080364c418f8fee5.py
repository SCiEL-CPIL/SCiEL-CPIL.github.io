"""Cost-aware acquisition for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = 320.0  # wall-clock safety margin (300s per eval, 330s total kill)
_START_TIME = None


def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def _ei_per_cost(mu, sigma, f_best, cost):
    """Expected improvement per unit cost."""
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return ei / np.maximum(cost, 1e-10)


def _ucb_per_cost(mu, sigma, f_best, cost, beta=1.96):
    """UCB per unit cost."""
    ucb = mu + beta * sigma
    # We want to minimize, so higher UCB (less negative) means more uncertain/high potential
    # For minimization, we prefer low mu and high sigma (exploration)
    # UCB for minimization: mu - beta*sigma is lower bound; we want to explore where lower bound is low
    # Actually for minimization, the "upper confidence bound" in the sense of "most promising"
    # is: we want points where the true value could be very low.
    # Standard UCB for minimization: score = mu - beta * sigma (lower is better)
    # But we need higher score = evaluate next, so: score = -(mu - beta*sigma) = -mu + beta*sigma
    # Per cost: (-mu + beta*sigma) / cost
    return (-mu + beta * sigma) / np.maximum(cost, 1e-10)


def _mixed_acquisition(mu, sigma, f_best, cost):
    """
    Mixed acquisition combining EI and UCB-like exploration, weighted by cost.
    Uses adaptive beta based on how close we are to finding the minimum.
    """
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # EI component
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei_per_cost = ei / cost_safe
    
    # Exploration component: reward high uncertainty where the mean is also promising
    # Use a form that balances exploitation (low mu) and exploration (high sigma)
    # Normalize mu relative to f_best
    mu_norm = np.maximum(f_best - mu, 0)  # improvement potential
    sigma_norm = sigma_safe
    
    # Combined: weight EI with exploration bonus
    # When f_best is close to 0 (good), reduce exploration
    # When f_best is far from 0, increase exploration
    gap = f_best  # since min is 0, gap = f_best - 0 = f_best
    gap_clipped = np.clip(gap, 0.01, 100)
    
    # Adaptive beta: more exploration when far from optimum
    beta = 1.0 + 2.0 * np.log1p(gap_clipped)
    beta = np.clip(beta, 1.0, 4.0)
    
    # UCB-like score for minimization (higher is better for selection)
    ucb_score = -mu + beta * sigma_safe
    ucb_per_cost = ucb_score / cost_safe
    
    # Blend: use EI primarily, but add exploration when EI is low
    # Find where EI is weak relative to UCB
    max_ei = np.max(ei_per_cost) if np.max(ei_per_cost) > 0 else 1.0
    max_ucb = np.max(ucb_per_cost) if np.max(ucb_per_cost) > 0 else 1.0
    
    # Normalize both to [0, 1] range for blending
    ei_norm = ei_per_cost / (max_ei + 1e-10)
    ucb_norm = ucb_per_cost / (max_ucb + 1e-10)
    
    # Weight: when EI signal is strong, rely on it; when weak, use UCB
    # Use a soft blend
    ei_strength = np.clip(ei_norm, 0, 1)
    w_ei = 0.7 * ei_strength + 0.3
    w_ucb = 1.0 - w_ei
    
    score = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Scale back to original magnitude for consistency
    return score * max_ei


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value so far
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    global _START_TIME
    
    # Deadline guard
    if _START_TIME is None:
        _START_TIME = time.time()
    
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Return a safe fallback: just use EI per cost
        sigma_safe = np.maximum(sigma, 1e-10)
        cost_safe = np.maximum(cost, 1e-10)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return ei / cost_safe
    
    # Ensure inputs are finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-10)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    sigma = np.maximum(sigma, 1e-10)
    cost = np.maximum(cost, 1e-10)
    
    # Use the mixed acquisition
    scores = _mixed_acquisition(mu, sigma, f_best, cost)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
