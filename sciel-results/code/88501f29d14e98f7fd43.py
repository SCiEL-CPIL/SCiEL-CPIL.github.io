"""Cost-aware acquisition function for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 285.0  # 285s safety margin (limit is 300s)


def _check_deadline():
    return time.time() > _DEADLINE


def _compute_ei(mu, sigma, f_best):
    """Compute Expected Improvement."""
    improvement = f_best - mu
    sigma_safe = np.maximum(sigma, 1e-9)
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _compute_ucb(mu, sigma, beta):
    """Compute Upper Confidence Bound (for minimization, we want low values)."""
    return mu - beta * sigma


def _compute_pi(mu, sigma, f_best):
    """Compute Probability of Improvement."""
    improvement = f_best - mu
    sigma_safe = np.maximum(sigma, 1e-9)
    z = improvement / sigma_safe
    pi = norm.cdf(z)
    return pi


def _compute_hedge(mu, sigma, f_best, cost):
    """
    Cost-aware hedged acquisition that combines EI, UCB, and PI.
    Uses adaptive weighting based on uncertainty and cost.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute base acquisition functions
    ei = _compute_ei(mu, sigma, f_best)
    pi = _compute_pi(mu, sigma, f_best)
    
    # Adaptive beta for UCB based on uncertainty distribution
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma)
    
    # Beta scales with uncertainty but is bounded
    beta = np.clip(0.5 + 0.3 * (sigma_std / (sigma_mean + 1e-9)), 0.5, 1.5)
    ucb = _compute_ucb(mu, sigma, beta)
    
    # Normalize each component to [0, 1] range for stable combination
    # EI normalization
    ei_max = np.max(ei) if np.max(ei) > 1e-9 else 1.0
    ei_norm = ei / ei_max
    
    # UCB normalization (lower is better, so we negate)
    ucb_neg = -ucb
    ucb_min = np.min(ucb_neg)
    ucb_max = np.max(ucb_neg)
    if ucb_max - ucb_min > 1e-9:
        ucb_norm = (ucb_neg - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.ones_like(ucb_neg)
    
    # PI is already in [0, 1]
    pi_norm = pi
    
    # Adaptive weights based on cost and uncertainty
    # High uncertainty regions get more exploration weight
    sigma_ratio = sigma / (sigma_mean + 1e-9)
    
    # Weight for EI: higher when uncertainty is moderate (not too low, not too high)
    w_ei = 0.4 + 0.2 * np.exp(-np.abs(sigma_ratio - 1.0))
    
    # Weight for UCB: higher when uncertainty is high
    w_ucb = 0.2 + 0.2 * np.clip(sigma_ratio, 0, 2) / 2
    
    # Weight for PI: baseline exploration
    w_pi = 0.2 + 0.1 * np.clip(sigma_ratio, 0, 2) / 2
    
    # Normalize weights to sum to 1
    w_sum = w_ei + w_ucb + w_pi
    w_ei = w_ei / w_sum
    w_ucb = w_ucb / w_sum
    w_pi = w_pi / w_sum
    
    # Combine normalized components
    score = w_ei * ei_norm + w_ucb * ucb_norm + w_pi * pi_norm
    
    # Apply cost-aware penalty: divide by cost to prefer cheaper evaluations
    score = score / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Check deadline
    if _check_deadline():
        # Return a simple fallback: prefer low cost and low mu
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and bounded
    sigma = np.clip(sigma, 1e-6, 1e6)
    
    # Ensure cost is positive and bounded
    cost = np.clip(cost, 1e-6, 1e6)
    
    # Compute the hedged acquisition score
    score = _compute_hedge(mu, sigma, f_best, cost)
    
    # Add a tiny noise to break ties and ensure diversity
    # This helps avoid getting stuck in local optima
    rng = np.random.default_rng(42)
    score += rng.normal(0, 1e-8, size=score.shape)
    
    # Final check for finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
