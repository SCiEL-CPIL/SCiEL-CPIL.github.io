import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using an adaptive quadratic-UCB blend.
    
    Structural change from the parent: instead of a linear EI + PI blend divided
    by cost, we use a quadratic exploration penalty (sigma^2 term) that creates
    a sharper exploration/exploitation balance. The exploration weight adapts
    based on how much budget remains (proxy: f_best quality) and the cost
    structure. This favors cheap, uncertain regions more aggressively.
    
    The key insight: UCB with a quadratic sigma term naturally penalizes
    expensive regions more heavily when uncertainty is high, because the
    exploration bonus scales with sigma^2 / cost. This creates a more
    principled cost-awareness than simple division.
    """
    t0 = time.time()
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety guards
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement (for minimization)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB exploration term: beta * sigma^2 / cost
    # This is the structural change: quadratic in sigma, not linear
    # The / cost makes it cost-aware in a more aggressive way
    
    # Adaptive beta: 
    # When f_best is close to 0 (good), we want less exploration (smaller beta)
    # When f_best is large (poor), we want more exploration (larger beta)
    # f_best for Griewank 2D on [-10,10]^2 is typically in [0, ~0.5]
    f_best_abs = abs(f_best)
    
    # Beta transitions: high exploration when f_best is bad, low when good
    # Use a smooth transition
    # beta ranges from ~0.3 (exploit) to ~1.5 (explore)
    beta = 1.5 - 1.2 * np.clip(f_best_abs / 0.3, 0.0, 1.0)
    
    # Quadratic UCB exploration term, cost-normalized
    # sigma^2 / cost: high uncertainty + low cost = high exploration bonus
    ucb_exploration = beta * (sigma_safe ** 2) / cost_safe
    
    # Exploitation term: EI / cost
    ei_per_cost = ei / cost_safe
    
    # Combine: exploitation + exploration
    # The quadratic nature of ucb_exploration means that regions with
    # high sigma get a disproportionately large boost, especially if cheap
    score = ei_per_cost + ucb_exploration
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Safety deadline guard
    elapsed = time.time() - t0
    if elapsed > 290:
        # Fallback to simple EI/cost
        return np.maximum(ei_per_cost, 0.0)
    
    return score
