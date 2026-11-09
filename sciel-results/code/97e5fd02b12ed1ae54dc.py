# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_DEADLINE = None
def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 290  # 290s safety margin from 300s limit
    return _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Levy 2D.
    
    Uses a structural change: Probability of Improvement (PoI) with 
    cost-aware variance exploration.
    
    Score = PoI / cost + beta * (sigma / cost)
    
    PoI measures the probability that f(x) < f_best.
    The variance term encourages exploration in high-uncertainty regions,
    but is discounted by the evaluation cost.
    
    Parameters:
    - mu: (2000,) posterior mean in objective units
    - sigma: (2000,) posterior std in objective units  
    - f_best: scalar, lowest observed value
    - cost: (2000,) evaluation cost for each candidate
    
    Returns:
    - (2000,) scores, higher = evaluate next
    """
    t_start = time.time()
    deadline = _get_deadline()
    
    n = len(mu)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # 1. Probability of Improvement (PoI)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # We add a small epsilon to sigma to avoid division by zero
    z = (f_best - mu) / (sigma + 1e-9)
    poi = norm.cdf(z)
    
    # 2. Cost-aware PoI
    # We want to maximize the probability of improvement per unit cost.
    poi_per_cost = poi / cost
    
    # 3. Exploration Term
    # We encourage visiting points with high uncertainty (sigma).
    # However, if the cost is high, the value of that uncertainty is lower.
    # We also want to ensure we don't explore too much if f_best is already very good (close to 0).
    # Since Levy 2D min is 0, if f_best is close to 0, we should exploit more.
    # Let's scale the exploration weight by the gap to the global minimum (0).
    # gap = f_best - 0 = f_best. If f_best is large, we explore more. If f_best is small, we exploit more.
    # However, f_best is in objective units. Levy 2D can be large initially.
    # A simpler structural change is just using sigma/cost as a pure exploration proxy.
    # Let's use a fixed beta that balances exploration and exploitation.
    # Based on typical BO scales, beta around 0.1 to 0.5 is common for sigma terms.
    # Since we are dividing by cost, and cost is in [1, 2], the term is bounded.
    
    # Let's try a dynamic beta: if f_best is high (bad), explore more.
    # f_best is the current best value.
    # If f_best is > 1, we are far from optimum, explore more.
    # If f_best is < 0.1, we are close, exploit more.
    # Let's map f_best to a weight.
    # weight = 1 / (1 + f_best) ? No, f_best can be 0.
    # Let's just use a constant beta for simplicity and robustness, as PoI is already sensitive to mu.
    beta = 0.1
    
    exploration_score = beta * (sigma / cost)
    
    # Combine
    score = poi_per_cost + exploration_score
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check
    if time.time() > deadline - 5:  # 5s safety margin
        # Return a simple fallback: prefer low mu and low cost
        fallback = (f_best - mu) / (cost + 1e-9)
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    return score
# EVOLVE-BLOCK-END
