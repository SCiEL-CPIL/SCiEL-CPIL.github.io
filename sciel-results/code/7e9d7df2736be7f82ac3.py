import numpy as np
from scipy.stats import norm
import time

_START_TIME = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Effective Probability of Improvement (PoI).
    
    Strategy:
    1. Compute Probability of Improvement: P(f(x) < f_best) = Phi((f_best - mu) / sigma).
       Unlike EI, PoI is bounded [0, 1] and does not amplify noise via the sigma * pdf(z) term.
    2. Normalize by absolute cost. The cost model is 1 + mean(x_scaled).
       Dividing by absolute cost explicitly rewards points that are "cheap" to evaluate.
       This is more stable than marginal cost normalization which can blow up when 
       min_cost is close to other costs.
    3. Add a small, adaptive exploration bonus based on uncertainty (sigma) to prevent
       the search from collapsing into a single low-cost, low-uncertainty local minimum 
       too early. The bonus is scaled relative to the best PoI score to maintain 
       dimensionless consistency.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    elapsed = time.time() - _START_TIME
    
    # Safety guard: if we are close to the deadline, return a simple, fast score
    if elapsed > 325.0:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        z_clipped = np.clip(z, -10, 10)
        poi = norm.cdf(z_clipped)
        cost_safe = np.maximum(cost, 1e-9)
        score = poi / cost_safe
        score = np.where(np.isfinite(score), score, 0.0)
        return score

    # 1. Compute Probability of Improvement (PoI)
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical saturation in CDF
    z_clipped = np.clip(z, -10, 10)
    poi = norm.cdf(z_clipped)
    
    # 2. Cost-normalize
    cost_safe = np.maximum(cost, 1e-9)
    
    # Cost-Effective PoI: Maximize probability of improvement per unit cost
    # This favors points with high likelihood of beating f_best AND low evaluation cost
    base_score = poi / cost_safe
    
    # 3. Adaptive Exploration Bonus
    # In the early stages, sigma is high everywhere. As we explore, sigma drops.
    # We want to encourage moving to areas with high sigma (uncertainty) if the 
    # exploitation signal (base_score) is not overwhelmingly strong.
    
    # Normalize sigma to [0, 1] range for this batch
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    if sigma_max > sigma_min:
        sigma_norm = (sigma_safe - sigma_min) / (sigma_max - sigma_min)
    else:
        sigma_norm = np.ones_like(sigma_safe)
        
    # Scale the exploration bonus.
    # We use a small coefficient (0.1) to ensure exploitation dominates, 
    # but exploration can break ties or push towards unexplored regions 
    # when base scores are similar.
    # The bonus is proportional to the uncertainty.
    max_base = np.max(base_score)
    
    # If base scores are very low (early or late stage where improvement is unlikely),
    # we might want more exploration, but let's keep it simple:
    # Bonus is 0.1 * max_base * sigma_norm
    # This means max exploration bonus is 0.1 * max_base.
    # So exploration can at most add 10% to the best exploitation score.
    
    if max_base > 1e-12:
        exploration_bonus = 0.1 * max_base * sigma_norm
    else:
        exploration_bonus = np.zeros_like(sigma_norm)
        
    score = base_score + exploration_bonus
    
    # Ensure finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
