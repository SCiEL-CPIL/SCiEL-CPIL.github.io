"""Rewritten program using global budget-aware cost penalty strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # Seconds to spare

def _is_deadline_exceeded():
    """Check if we are close to the global time limit."""
    if time.time() - _START_TIME > (300.0 - _DEADLINE_MARGIN):
        return True
    return False

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using global budget-aware scaling.
    Computes normalized expected improvement and subtracts a penalty term
    proportional to the current cost relative to the mean cost of the batch.
    
    Parameters:
    mu: array-like, mean predictions from GP
    sigma: array-like, standard deviation predictions from GP
    f_best: float, best observed value so far
    cost: array-like, cost of evaluating each candidate
    
    Returns:
    scores: array of finite scores, higher is better
    """
    # Safety check: if time is running out, return a safe random-like score
    if _is_deadline_exceeded():
        return np.zeros_like(mu, dtype=float)

    # Ensure inputs are numpy arrays for vectorization
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Numerical stability
    epsilon = 1e-8
    sigma_safe = np.clip(sigma, epsilon, None)
    cost_safe = np.clip(cost, epsilon, None)
    
    # Estimate progress based on f_best to adapt exploration/exploitation balance
    f_min_expected = -3.86278
    f_max_expected = 3.86
    
    # Normalize f_best to a progress metric [0, 1]
    # If f_best is very bad (high), progress is low -> explore more.
    # If f_best is very good (low), progress is high -> exploit more.
    progress = (f_max_expected - f_best) / (f_max_expected - f_min_expected)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Adaptive Beta:
    # Start with high exploration (beta ~ 2.0) and decay to low exploration (beta ~ 0.5).
    beta_max = 2.0
    beta_min = 0.5
    beta = beta_max - (beta_max - beta_min) * progress
    
    # Compute the base acquisition score (Linear UCB for minimization)
    # Score = (f_best - mu + beta * sigma)
    # This is positive when we expect improvement (mu < f_best) or high uncertainty
    base_score = f_best - mu + beta * sigma_safe
    
    # Global budget-aware scaling:
    # Instead of dividing by individual cost, we compute a normalized score
    # and subtract a penalty based on how expensive this candidate is relative
    # to the mean cost of the current batch of 2000 candidates.
    
    # Compute mean and std of costs in the batch
    cost_mean = np.mean(cost_safe)
    cost_std = np.std(cost_safe)
    cost_std_safe = max(cost_std, epsilon)
    
    # Normalize cost to z-score: (cost - mean) / std
    # Positive z-score means more expensive than average, negative means cheaper.
    cost_zscore = (cost_safe - cost_mean) / cost_std_safe
    
    # Penalty term: proportional to the z-score of the cost
    # We want to penalize expensive candidates and reward cheap ones.
    # The penalty should be scaled by a factor that balances the magnitude of base_score.
    # Let's scale the penalty by the mean absolute base score to make it adaptive.
    mean_abs_score = np.mean(np.abs(base_score))
    if mean_abs_score < epsilon:
        mean_abs_score = epsilon
        
    # Penalty coefficient: how strongly we penalize expensive candidates
    # This can be adaptive based on progress: more penalty early on to save budget?
    # Or maybe constant? Let's start with a moderate constant.
    penalty_coeff = 0.5 * mean_abs_score
    
    # Final score = base_score - penalty_coeff * cost_zscore
    # If cost is above mean (z > 0), score decreases (penalty).
    # If cost is below mean (z < 0), score increases (bonus).
    scores = base_score - penalty_coeff * cost_zscore
    
    # Final safety: replace any non-finite values with 0
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

# EVOLVE-BLOCK-END
