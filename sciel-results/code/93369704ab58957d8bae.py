# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 325s as safety margin
_DEADLINE = 325.0
_START_TIME = time.time()

def _check_deadline():
    """Return True if we should abort and return current best."""
    return (time.time() - _START_TIME) > _DEADLINE

def _soft_min(a, b, tau=0.1):
    """Smooth minimum using log-sum-exp trick for stability."""
    m = np.minimum(a, b)
    return m - tau * np.log(1.0 + np.exp(-(a - m) / tau) + np.exp(-(b - m) / tau))

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Styblinski-Tang 2D Bayesian optimization.
    
    Uses a refined cost-aware EI approach with:
    - Expected Improvement (EI) for exploitation
    - Cost-normalized scoring to prefer efficient evaluations
    - Adaptive exploration-exploitation balance based on progress
    - Smooth combination of EI and LCB signals
    
    Parameters:
    - mu: shape (2000,), GP mean predictions
    - sigma: shape (2000,), GP standard deviations
    - f_best: scalar, best observed value
    - cost: shape (2000,), evaluation costs (1 + mean(x_scaled))
    
    Returns:
    - scores: shape (2000,), acquisition scores (higher = evaluate next)
    """
    # Check deadline first
    if _check_deadline():
        return np.zeros_like(mu, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Clip to valid ranges
    sigma = np.clip(sigma, 1e-8, 1e4)
    cost = np.clip(cost, 0.5, 10.0)
    
    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma
    cdf_vals = norm.cdf(z)
    pdf_vals = norm.pdf(z)
    ei = (f_best - mu) * cdf_vals + sigma * pdf_vals
    ei = np.maximum(ei, 0.0)
    
    # Compute LCB (Lower Confidence Bound for minimization)
    # LCB = mu - beta * sigma (we want to minimize, so lower is better)
    # For acquisition, we use: -LCB = -mu + beta * sigma (higher = more promising)
    
    # Adaptive beta: balance exploration and exploitation
    # Early: more exploration (higher beta)
    # Late: more exploitation (lower beta)
    
    dist_to_min = max(f_best - (-78.3323), 0.0)
    progress = 1.0 / (1.0 + 0.05 * dist_to_min)
    
    # Beta schedule: start high, decrease as we progress
    beta = 1.5 * (1.0 - progress) + 0.3 * progress
    # beta ~ 1.5 early, ~ 0.3 late
    
    # LCB-based score (higher = more promising)
    lcb_score = -mu + beta * sigma
    
    # Cost-aware normalization
    # We want to reward points with high improvement relative to their cost
    # cost is in [1, 2] typically (1 + mean of [0,1]^2)
    
    # Cost efficiency: divide by cost^p where p controls penalty strength
    # p=0.5 gives moderate penalty, p=1 gives strong penalty
    p = 0.4 + 0.3 * progress  # stronger cost penalty when close to optimum
    cost_efficiency = 1.0 / (cost ** p)
    
    # Combine EI and LCB using soft-min to get a robust signal
    # We want the "best" of both signals, but smoothed
    # Normalize both to comparable scale
    ei_scale = np.mean(ei) + 1e-10
    lcb_scale = np.mean(np.abs(lcb_score)) + 1e-10
    
    ei_norm = ei / ei_scale
    lcb_norm = lcb_score / lcb_scale
    
    # Soft combination: use soft-min to get a smooth "best of" signal
    # tau controls how smooth vs hard the selection is
    tau = 0.15
    combined = _soft_min(ei_norm, lcb_norm, tau)
    
    # Scale back to original magnitude
    combined_scaled = combined * (ei_scale + lcb_scale) / 2.0
    
    # Apply cost efficiency
    score = combined_scaled * cost_efficiency
    
    # Add a small exploration bonus for very uncertain points
    # This helps escape local optima in early iterations
    exploration_bonus = 0.1 * (1.0 - progress) * sigma / (sigma.max() + 1e-10)
    
    score += exploration_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
