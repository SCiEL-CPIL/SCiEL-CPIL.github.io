"""Cost-aware acquisition for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = _START_TIME + 295  # safety margin before 300s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition combining Expected Improvement (EI) with an
    adaptive exploration bonus to handle Rastrigin's multimodality.
    
    The score is: (EI + adaptive_exploration) / cost^0.5
    
    The adaptive exploration bonus uses a beta that increases when f_best
    is still far from the known minimum (0), encouraging exploration early
    on, and decreases as we get closer, favoring exploitation.
    
    Args:
        mu: (2000,) GP mean predictions in objective units.
        sigma: (2000,) GP std predictions in objective units.
        f_best: float, best observed (lowest) value.
        cost: (2000,) evaluation cost per candidate.
    
    Returns:
        scores: (2000,) finite scores, higher = pick next.
    """
    if time.time() > _DEADLINE:
        # Fallback: prefer cheap points with some uncertainty
        return 1.0 / (np.maximum(cost, 1e-9) + 1e-9)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Standardize: z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Clip negative EI to zero (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive beta: start high (exploration) when f_best is large,
    # decrease as we approach the minimum (exploitation)
    # f_best starts around 10-20 for Rastrigin, minimum is 0
    # Use a smooth decay: beta = 2.0 * exp(-f_best / 5.0) + 1.0
    # When f_best = 20: beta = 2*exp(-4) + 1 ≈ 1.03
    # When f_best = 5: beta = 2*exp(-1) + 1 ≈ 1.74
    # When f_best = 1: beta = 2*exp(-0.2) + 1 ≈ 1.67
    # When f_best = 0: beta = 2*1 + 1 = 3.0 (but we're done)
    # Actually, let's use a simpler scaling that's more effective:
    # beta = 1.5 + 1.5 * min(f_best / 10.0, 1.0)
    # When f_best = 20: beta = 1.5 + 1.5*1.0 = 3.0
    # When f_best = 10: beta = 1.5 + 1.5*1.0 = 3.0
    # When f_best = 5: beta = 1.5 + 1.5*0.5 = 2.25
    # When f_best = 1: beta = 1.5 + 1.5*0.1 = 1.65
    # When f_best = 0: beta = 1.5
    
    beta = 1.5 + 1.5 * min(f_best / 10.0, 1.0)
    
    # Exploration bonus: beta * sigma
    exploration_bonus = beta * sigma_safe
    
    # Combine EI and exploration
    # Use a weighted combination to balance exploitation and exploration
    score = (ei + exploration_bonus) / (cost_safe ** 0.5)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
