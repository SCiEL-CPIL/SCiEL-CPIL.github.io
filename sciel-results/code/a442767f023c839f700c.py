# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s wall-clock.
_t_start = None
_DEADLINE = 310.0


def acquisition(mu, sigma, f_best, cost):
    global _t_start
    if _t_start is None:
        _t_start = time.time()

    elapsed = time.time() - _t_start
    if elapsed > _DEADLINE:
        # Emergency: return a simple valid score
        eps = 1e-12
        z = (f_best - mu) / (sigma + eps)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    eps = 1e-12

    # Standard EI
    z = (f_best - mu) / (sigma + eps)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Cost-aware: EI per unit cost
    ei_cost = ei / (cost + eps)

    # Structural change: instead of a small exploration bonus,
    # use a more aggressive exploration strategy via a UCB-like term
    # weighted by uncertainty relative to the range of predictions.
    # This helps escape rastrigin's many local minima.
    
    # UCB for minimization: we want to explore where sigma is high
    # relative to the spread of mu values
    mu_range = np.ptp(mu)
    if mu_range < eps:
        mu_range = 1.0
    
    # Normalize sigma to [0,1] range for consistent exploration pressure
    sigma_norm = sigma / (sigma.max() + eps)
    
    # Exploration weight: adaptive - more exploration early, less later
    # Use the ratio of current best to typical function values as a proxy
    # for how much progress has been made
    # Since we don't know the scale, use a fixed moderate exploration
    # that's structured differently from before
    
    # Use a product structure: exploration bonus that's strong when
    # sigma is high AND mu is not already very good
    exploration_mask = sigma_norm * (1.0 - norm.cdf((mu - f_best) / (sigma + eps)))
    
    # Combine: primary EI/cost with secondary exploration term
    # The exploration term is NOT simply sigma/cost but a more
    # structured quantity that accounts for the GP's belief
    score = ei_cost + 0.3 * exploration_mask / (cost + eps)
    
    # Also add a small penalty for high cost to make the tradeoff sharper
    # This is different from just dividing EI by cost - it adds an
    # explicit cost sensitivity that's separable from the improvement estimate
    cost_penalty = 0.1 * (cost - cost.min()) / (cost.max() - cost.min() + eps)
    score = score - cost_penalty * 0.05
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
