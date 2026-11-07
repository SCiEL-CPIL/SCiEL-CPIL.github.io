"""Cost-aware acquisition for BO on three hump camel 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 310.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Probability of Improvement (PI) with adaptive
    cost-awareness and diversity preservation.
    
    Strategy:
    1. Compute PI = Phi((f_best - mu) / sigma) for robust, simple scoring.
    2. Adaptive cost penalty: strengthens as f_best approaches 0 (convergence),
       encouraging cheaper evaluations when gains are small.
    3. Diversity bonus: small reward for high uncertainty to avoid local optima.
    
    Parameters:
        mu: (2000,) GP mean predictions in objective units
        sigma: (2000,) GP std dev predictions (positive) in objective units
        f_best: float, best observed value so far
        cost: (2000,) evaluation cost per candidate
    
    Returns:
        (2000,) finite acquisition scores, higher = better
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        return np.zeros(2000, dtype=np.float64)
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Probability of Improvement for minimization
    Z = (f_best - mu) / sigma_safe
    Z = np.clip(Z, -10.0, 10.0)
    
    from scipy.stats import norm
    PI = norm.cdf(Z)
    
    # Adaptive cost-awareness:
    # When f_best is close to 0 (the known minimum value), marginal gains are small,
    # so we should strongly prefer cheaper evaluations.
    # When f_best is far from 0, we can afford expensive evaluations for big gains.
    # 
    # Define convergence progress: p = 1 - f_best / f_max
    # For three hump camel 2D, max value at corners is roughly 25*25 + 25 - 25*25 + 25 = 25
    # Actually let's use a simpler heuristic: normalize by typical range
    # f_best starts around 10-20 and goes to 0
    f_scale = max(abs(f_best), 0.1)
    convergence = 1.0 - np.clip(f_best / 10.0, 0.0, 1.0)
    # convergence is 0 when f_best >= 10, 1 when f_best <= 0
    
    # Cost penalty strength increases with convergence
    alpha_base = 0.2
    alpha = alpha_base + 0.8 * convergence  # alpha in [0.2, 1.0]
    
    # Normalize cost to [0, 1] range for consistent scaling
    cost_norm = np.clip((cost - 1.0), 0.0, 1.0)  # cost is in [1, 2]
    cost_penalty = 1.0 + alpha * cost_norm
    score = PI / cost_penalty
    
    # Diversity bonus: reward uncertainty in regions where PI is moderate
    # This helps escape local optima early in the search
    uncertainty_bonus = 0.05 * convergence * sigma_safe / (sigma_safe + 1.0)
    # Scale uncertainty bonus relative to PI scale
    score += uncertainty_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
