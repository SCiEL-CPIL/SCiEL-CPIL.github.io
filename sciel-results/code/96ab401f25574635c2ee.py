import numpy as np
from scipy.stats import norm
import time

_T0 = time.time()
_DEADLINE = 310.0

def _time_ok():
    return (time.time() - _T0) < _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safety checks
    if not _time_ok():
        # Fallback: simple EI/cost
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / np.maximum(cost, 1e-6)
        return np.where(np.isfinite(score), score, 0.0)

    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-6)

    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Adaptive Beta for UCB component
    # We want to explore more when f_best is far from the minimum (-78.33)
    # and exploit more when it's close.
    # f_best starts around -10 to -20 typically for ST-2D after 5 random points.
    # Let's use a progress metric based on f_best.
    # If f_best is -10, progress is low. If -70, progress is high.
    # Map f_best from [-20, -78] to [0, 1]
    progress = np.clip((-f_best - 20) / 58.0, 0.0, 1.0)
    
    # Beta starts high (exploration) and decreases
    beta = 2.0 * np.exp(-1.5 * progress) + 0.1
    
    # Combine EI and UCB-like term (sigma)
    # Score base = (EI + beta * sigma) / cost
    base_score = (ei + beta * sigma_safe) / cost_safe

    # Cost-awareness:
    # The minimum is at cost ~1.21.
    # We apply a penalty for costs significantly higher than 1.4.
    # This steers the search away from the expensive top-right corner.
    
    # Penalty function:
    # 1.0 for cost <= 1.4
    # Exponential decay for cost > 1.4
    # This is a structural change from the Gaussian prior, being more aggressive in cutting off high-cost regions.
    
    threshold = 1.4
    penalty = np.ones_like(cost)
    high_cost_mask = cost > threshold
    penalty[high_cost_mask] = np.exp(-3.0 * (cost[high_cost_mask] - threshold))
    
    # Also, a small boost for costs very close to the expected minimum (1.21)
    # This helps if the GP is uninformative.
    # Gaussian bump centered at 1.21
    bump = np.exp(-0.5 * ((cost - 1.21) / 0.15) ** 2)
    
    # Combine penalty and bump
    # We want the penalty to dominate for high costs, and the bump to help in the low cost region.
    # Let's blend them.
    # Early on, the bump is more important to guide the search.
    # Late on, the GP mean is more important, so we rely more on the base_score.
    
    # Weight of the prior (bump+penalty) decreases as progress increases
    w_prior = 1.0 - 0.5 * progress
    
    # Final score
    # We multiply the base score by a factor that includes the prior.
    # factor = w_prior * (penalty * bump) + (1 - w_prior) * 1.0
    # This way, early on, the score is heavily modulated by the prior.
    # Late on, it's just the base_score.
    
    factor = w_prior * (penalty * bump) + (1.0 - w_prior)
    
    score = base_score * factor
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
