"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware LCB acquisition function for Bayesian optimization.
    Uses an adaptive exploration coefficient (beta) that decreases as f_best improves,
    normalized by evaluation cost.
    
    Score = (beta * sigma - mu) / cost
    """
    start_time = time.time()
    DEADLINE = 320.0  # Safety margin before 330s kill

    n = mu.shape[0]
    
    # Clamp sigma for numerical stability
    sigma_safe = np.clip(sigma, 1e-12, None)
    
    # Clamp cost for numerical stability
    cost_safe = np.maximum(cost, 1e-10)
    
    # Adaptive beta with logarithmic schedule:
    # - Uses log(1 + f_best) for smoother decay than sqrt
    # - Multiplier of 2.0 for stronger early exploration
    # - Floor at 1.0 ensures exploitation when f_best is near 0
    f_best_clamped = max(f_best, 0.0)
    beta = 1.0 + 2.0 * np.log1p(f_best_clamped)
    
    # Lower Confidence Bound value: mu - beta * sigma
    # We want to minimize this value.
    # Acquisition score (higher is better) is the negative of LCB divided by cost.
    # score = - (mu - beta * sigma) / cost = (beta * sigma - mu) / cost
    
    # Use a slightly stronger cost penalty: cost^0.8 instead of cost
    # This makes low-cost candidates slightly more attractive
    cost_weighted = np.power(cost_safe, 0.8)
    scores = (beta * sigma_safe - mu) / cost_weighted
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Fallback: simple cost-aware UCB (minimize mu + k*sigma)
        # This is extremely fast, so this branch is unlikely to be triggered
        ucb = mu + 1.5 * sigma_safe
        simple = -ucb / cost_safe
        return np.where(np.isfinite(simple), simple, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
