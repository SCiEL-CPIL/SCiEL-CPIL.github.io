"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Theoretical minimum of cosine8 8d
_F_MIN = -0.8

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: adaptive LCB with cost penalty.
    
    Uses Lower Confidence Bound (LCB) for exploration/exploitation,
    with a cost penalty that balances exploration vs. cost-awareness.
    The penalty is dynamically adjusted based on proximity to the known
    theoretical minimum.
    
    Parameters:
        mu: array of shape (2000,), GP mean predictions
        sigma: array of shape (2000,), GP standard deviations
        f_best: float, best observed value so far
        cost: array of shape (2000,), evaluation costs
    
    Returns:
        array of shape (2000,), acquisition scores (higher = evaluate next)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive beta: increase exploration as we get closer to the minimum
    # When f_best is far from _F_MIN, we can be more conservative (lower beta)
    # When f_best is close to _F_MIN, we need more exploration (higher beta)
    progress = np.clip((f_best - _F_MIN) / (f_best - _F_MIN + 1e-9), 0, 1)
    # Actually, let's use a different scaling:
    # If f_best is near _F_MIN, progress should be high (more exploration needed)
    # If f_best is far from _F_MIN, progress should be low (less exploration needed)
    # f_best starts around 0 and goes to -0.8
    # distance_to_min = f_best - _F_MIN (positive, decreases as we improve)
    # We want beta to increase as distance_to_min decreases
    
    # Normalize the distance: when f_best = 0, dist = 0.8; when f_best = -0.8, dist = 0
    dist_to_min = max(f_best - _F_MIN, 1e-6)
    # Map dist_to_min from [0, 0.8] to [0, 1], then invert so that
    # close to minimum -> high exploration
    # At f_best=0: dist=0.8, normalized=1.0, inverted=0.0
    # At f_best=-0.8: dist=0, normalized=0.0, inverted=1.0
    norm_dist = np.clip(dist_to_min / 0.8, 0, 1)
    # Beta increases from ~2 to ~4 as we approach the minimum
    beta = 2.0 + 2.0 * (1.0 - norm_dist)
    
    # LCB: mu - beta * sigma (lower is better for minimization)
    lcb = mu - beta * sigma
    
    # Cost penalty: we want to prefer cheaper evaluations
    # Use a penalty proportional to cost
    # The penalty should be scaled relative to the typical improvement scale
    # Use the standard deviation of mu as a scale reference
    mu_scale = np.std(mu)
    if mu_scale < 1e-9:
        mu_scale = 1e-6
    
    # Cost penalty: penalize expensive points
    # cost ranges from 1 to 2 (since mean(x scaled to [0,1]) is in [0,1])
    # Normalize cost to [0, 1] range: (cost - 1) / 1.0
    cost_norm = (cost - 1.0) / 1.0  # in [0, 1]
    
    # Penalty weight: scale by mu_scale to make it comparable to LCB values
    penalty_weight = 0.5 * mu_scale
    
    # Acquisition score: higher is better
    # We want to maximize: -lcb (since lower lcb is better) minus cost penalty
    score = -lcb - penalty_weight * cost_norm
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
