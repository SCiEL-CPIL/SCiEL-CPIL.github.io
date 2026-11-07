"""Cost-aware acquisition for rastrigin 2d BO loop."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s wall-clock.
_T0 = time.time()
_DEADLINE = 325.0  # safety margin of 5s before the 330s kill


def _deadline_exceeded():
    return (time.time() - _T0) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of rastrigin 2d.
    
    Combines Probability of Improvement (PI) and Expected Improvement (EI)
    with adaptive weighting based on the current best observed value.
    
    PI is robust to high-variance regions that promise large but unlikely improvements,
    which is critical for the multimodal Rastrigin function. EI provides a stronger
    exploitation signal when the search is close to the optimum.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard: if we're running out of time, return a simple fallback
    if _deadline_exceeded():
        # Fallback: simple expected improvement per cost
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -10, 10)
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return ei / (cost + 1e-9)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma and cost are positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)  # Clip to avoid numerical issues
    
    # Probability of Improvement: P(f(x) < f_best)
    pi = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * pi + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive weighting based on f_best
    # When f_best is close to 0 (minimum of rastrigin), rely more on EI for exploitation
    # When f_best is large, rely more on PI for robust exploration
    if f_best < 1.0:
        pi_weight = 0.3
        ei_weight = 0.7
    elif f_best < 10.0:
        pi_weight = 0.5
        ei_weight = 0.5
    else:
        pi_weight = 0.7
        ei_weight = 0.3
    
    # Normalize EI to be comparable with PI (PI is in [0,1], EI can be larger)
    # Use a soft normalization based on the scale of f_best
    f_scale = abs(f_best) + 1.0
    ei_normalized = ei / f_scale
    
    # Combine PI and EI
    score = pi_weight * pi + ei_weight * ei_normalized
    
    # Cost-awareness: divide by cost
    score = score / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
