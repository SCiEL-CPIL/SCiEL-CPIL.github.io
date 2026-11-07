import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290  # safety margin before 300s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Probability of Improvement
    (PoI) and Expected Improvement (EI) with adaptive exploration for Rastrigin-2D.
    
    The strategy combines:
    1. PoI for robust exploitation (probabilistic view of improvement)
    2. EI for stronger signal when improvement is likely
    3. Adaptive exploration bonus that decays with cost (budget) to balance
       exploration/exploitation over the course of the optimization
    4. Cost-aware normalization to prefer cheaper evaluations
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP std predictions
        f_best: float - best observed value (lowest)
        cost: array of shape (2000,) - evaluation cost per candidate
    
    Returns:
        scores: array of shape (2000,) - finite scores, higher = pick next
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback: simple cost-aware exploitation
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score for improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    Phi = norm.cdf(z)
    
    # Expected Improvement for minimization
    phi = norm.pdf(z)
    ei = (f_best - mu) * Phi + sigma_safe * phi
    ei = np.maximum(ei, 0.0)
    
    # Adaptive exploration coefficient
    # Use mean cost as a proxy for progress through the budget
    mean_cost = np.mean(cost)
    # Start with higher exploration, decay towards exploitation
    # At cost ~5 (early), kappa ~2.0; at cost ~30 (late), kappa ~0.5
    kappa = 2.0 - 0.05 * (mean_cost - 5.0)
    kappa = np.clip(kappa, 0.5, 2.5)
    
    # Hybrid score: blend PoI and EI with exploration bonus
    # PoI gives a bounded [0,1] signal; EI can be larger
    # Normalize EI by a typical scale to make it comparable to PoI
    # Use the mean EI as a scale reference
    ei_scale = np.mean(ei) + 1e-9
    ei_normalized = ei / ei_scale
    
    # Blend: 50% PoI + 50% normalized EI
    base_score = 0.5 * Phi + 0.5 * ei_normalized
    
    # Exploration bonus: reward uncertainty in regions not yet well-explored
    # This helps escape local minima in Rastrigin
    exploration_bonus = kappa * sigma_safe / (np.max(sigma) + 1e-9)
    
    # Combine base score with exploration bonus
    # Weight exploration less as we progress
    exploration_weight = 0.3 * (1.0 - np.clip((mean_cost - 5.0) / 25.0, 0.0, 1.0))
    score = base_score + exploration_weight * exploration_bonus
    
    # Cost-aware normalization: divide by cost to prefer cheaper points
    # Use a softened cost penalty: cost^0.7 instead of cost^1.0
    cost_penalty = cost_safe ** 0.7
    score = score / cost_penalty
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
