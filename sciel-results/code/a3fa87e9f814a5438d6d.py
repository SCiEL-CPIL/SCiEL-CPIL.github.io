"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _cost_normalized_ucb(mu, sigma, f_best, cost, beta=1.0):
    """
    Cost-normalized UCB acquisition.
    Higher scores are better (we want to minimize the objective).
    For minimization, we want low mu and high sigma.
    UCB for minimization: mu - beta * sigma
    Normalize by cost to prefer cheaper evaluations.
    """
    sigma_safe = np.maximum(sigma, 1e-8)
    
    ucb = f_best - mu + beta * sigma_safe
    # Use log-cost normalization for smoother penalty
    cost_safe = np.maximum(cost, 1.0)
    score = ucb / cost_safe
    
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score


def _cost_normalized_ei(mu, sigma, f_best, cost, xi=0.0):
    """
    Cost-normalized Expected Improvement.
    """
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # EI for minimization
    imp = f_best - mu - xi
    z = imp / sigma_safe
    ei = imp * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize by cost (log-smoothed)
    cost_safe = np.maximum(cost, 1.0)
    score = ei / cost_safe
    
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score


def _hybrid_acquisition(mu, sigma, f_best, cost):
    """
    Hybrid acquisition combining UCB and EI with adaptive weighting.
    Uses robust sigma normalization and adaptive beta.
    """
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Adaptive beta: higher exploration early, lower later
    # Use the spread of mu relative to f_best as a proxy for progress
    mu_range = np.max(mu) - np.min(mu)
    if mu_range > 1e-10:
        # If mu spread is large relative to f_best, we're still exploring
        progress = 1.0 - np.clip((f_best - np.min(mu)) / (mu_range + 1e-10), 0.0, 1.0)
    else:
        progress = 0.5
    
    # Beta decreases as we make progress: from 2.0 to 0.5
    beta = 0.5 + 1.5 * (1.0 - progress)
    
    # Adaptive xi: small exploitation incentive
    xi = 0.01 * max(0.0, f_best)
    
    # Compute both scores
    ucb_score = _cost_normalized_ucb(mu, sigma, f_best, cost, beta=beta)
    ei_score = _cost_normalized_ei(mu, sigma, f_best, cost, xi=xi)
    
    # Robust adaptive weighting using percentile instead of max
    sigma_p95 = np.percentile(sigma_safe, 95)
    sigma_p5 = np.percentile(sigma_safe, 5)
    sigma_range = sigma_p95 - sigma_p5
    
    if sigma_range > 1e-10:
        sigma_norm = np.clip((sigma_safe - sigma_p5) / sigma_range, 0.0, 1.0)
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Smooth S-curve weighting for better transition
    # w_ucb goes from ~0.3 to ~0.7 as sigma_norm goes 0 to 1
    w_ucb = 0.3 + 0.4 * (1.0 / (1.0 + np.exp(-4.0 * (sigma_norm - 0.5))))
    w_ei = 1.0 - w_ucb
    
    score = w_ucb * ucb_score + w_ei * ei_score
    
    # Add a small exploitation bonus for points very close to f_best
    # This helps converge when we're near the optimum
    close_mask = (f_best - mu) < 0.01 * max(abs(f_best), 1e-8)
    if np.any(close_mask):
        bonus = 0.1 * np.exp(-50.0 * (f_best - mu) / max(abs(f_best), 1e-8))
        score = score + bonus
    
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean predictions from the GP.
    sigma : ndarray, shape (2000,)
        Standard deviation predictions from the GP.
    f_best : float
        Best observed function value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    start_time = time.time()
    deadline = 290.0  # Safety margin before 300s limit
    
    # Validate inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    if mu.shape != (2000,) or sigma.shape != (2000,) or cost.shape != (2000,):
        raise ValueError("Input shapes must be (2000,)")
    
    # Handle edge cases
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(cost)):
        return np.ones(2000, dtype=np.float64)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1.0)
    
    # Use hybrid acquisition
    scores = _hybrid_acquisition(mu, sigma, f_best, cost)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline:
        pass
    
    # Final validation: ensure all scores are finite
    if not np.all(np.isfinite(scores)):
        finite_mask = np.isfinite(scores)
        if np.any(finite_mask):
            default_val = np.median(scores[finite_mask])
        else:
            default_val = 0.0
        scores = np.where(np.isfinite(scores), scores, default_val)
    
    return scores
# EVOLVE-BLOCK-END
