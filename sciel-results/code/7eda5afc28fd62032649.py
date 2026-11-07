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
    Cost-aware acquisition function combining Expected Improvement (EI)
    with a robust exploration term for multimodal Rastrigin 2D.
    
    Uses adaptive weighting between exploitation (EI) and exploration
    (uncertainty) based on the current best observed value. The cost
    structure (1 + mean(x_scaled)) is handled by dividing the score
    by the evaluation cost, favoring cheaper points.
    
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
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -10, 10)
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return ei / (cost + 1e-9)
    
    # Ensure inputs are numpy arrays with safe types
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
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)  # Clip to avoid numerical issues
    
    # Probability of Improvement: P(f(x) < f_best)
    pi = norm.cdf(z)
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    ei = (f_best - mu) * pi + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive weighting based on f_best
    # When f_best is close to 0 (minimum of rastrigin), rely more on EI for exploitation
    # When f_best is large, rely more on exploration to escape local minima
    if f_best < 1.0:
        pi_weight = 0.2
        ei_weight = 0.8
    elif f_best < 10.0:
        pi_weight = 0.4
        ei_weight = 0.6
    else:
        pi_weight = 0.6
        ei_weight = 0.4
    
    # Normalize EI to be comparable with PI (PI is in [0,1], EI can be larger)
    # Use a soft normalization based on the scale of f_best
    f_scale = abs(f_best) + 1.0
    ei_normalized = ei / f_scale
    
    # Exploration term: relative uncertainty
    exploration = sigma_safe / f_scale
    
    # Combine EI, PI, and exploration
    # EI provides strong exploitation signal
    # PI provides robust exploration in high-uncertainty regions
    # Exploration term maintains search pressure in multimodal landscapes
    score = ei_weight * ei_normalized + pi_weight * pi + 0.1 * exploration
    
    # Cost-awareness: divide by cost to penalize expensive evaluations
    score = score / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
