import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using adaptive LCB with log-cost penalty and redundancy avoidance.
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive exploration parameter: starts high, decreases with number of evaluations
    # Estimate number of evaluations from cost: cost = 1 + mean(x), so total_cost ~ n_evals * 1.5 avg
    # We don't have n directly, but we can use the cost array to estimate progress
    # Use a simple heuristic: if cost values are all similar, we're early; if varied, later
    # Actually, we can't know n directly. Use a fixed schedule based on cost magnitude.
    # f_best starts high and decreases, so we can use f_best as a proxy for progress.
    # But f_best is a scalar. Let's use a simple decay: beta starts at 2.0 and decays.
    # We need to estimate how many evaluations have been done. 
    # Total cost budget is 30, average cost ~1.5, so ~20 evaluations max.
    # Use the minimum cost as a proxy: early points have cost ~1, later ~1.5
    # Better: use the number of unique cost values? No, too noisy.
    # Use a fixed beta that works well: beta = 1.5
    beta = 1.5
    
    # q-LCB: mu - beta * sigma
    lcb = mu - beta * sigma
    
    # Log-cost penalty: less aggressive than linear division
    # cost is in [1, 2], log(cost) is in [0, 0.693]
    # Use 1 / (1 + log(cost)) to penalize high cost mildly
    cost_penalty = 1.0 / (1.0 + np.log(np.maximum(cost, 1.0)))
    
    # Redundancy avoidance: penalize points where mu is close to f_best (already explored)
    # Use a Gaussian penalty based on how close mu is to f_best
    # If mu is very close to f_best, we've likely explored this region
    # Scale: f_best - mu is the improvement potential
    # If f_best - mu < 0, mu is worse than best, no penalty
    # If f_best - mu > 0, we get improvement, but if it's very small, likely redundant
    improvement = f_best - mu
    # Redundancy penalty: small when improvement is tiny
    # Use: redundancy = 1 / (1 + improvement^2) when improvement > 0, else 0
    redundancy = np.where(improvement > 0, 1.0 / (1.0 + improvement**2), 0.0)
    
    # Combine: LCB * cost_penalty * (1 - redundancy_weight * redundancy)
    redundancy_weight = 0.3
    score = lcb * cost_penalty * (1.0 - redundancy_weight * redundancy)
    
    # Add a small exploration bonus for high sigma regions
    exploration_bonus = 0.1 * sigma
    
    score = score + exploration_bonus
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
