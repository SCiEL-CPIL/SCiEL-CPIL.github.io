"""Direct cost-aware acquisition using a log-domain stable EI/cost score."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 320s wall-clock safety margin (limit is 330s)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using UCB (Upper Confidence Bound).
    
    Structural change from EI to UCB: UCB directly rewards uncertainty
    (linear in sigma) rather than using the non-linear probabilistic EI.
    This better handles cost-aware settings by aggressively exploring
    high-uncertainty regions per unit cost.
    
    mu, sigma, f_best, cost: each shape (2000,)
    f_best: scalar (lowest observed value)
    cost: per-candidate cost = 1 + mean(x_scaled)
    
    Returns: 2000 finite scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero/negative values
    cost_safe = np.maximum(cost, 1e-9)
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # UCB: f_best - mu + beta * sigma
    # Linear uncertainty reward, directly proportional to sigma
    beta = 2.0
    ucb = f_best - mu + beta * sigma_safe
    
    # Cost-aware score: UCB / cost
    # Rewards high uncertainty and low predicted value per unit cost
    score = ucb / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
