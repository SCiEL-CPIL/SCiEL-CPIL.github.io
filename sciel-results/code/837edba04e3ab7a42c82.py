import numpy as np
from scipy.stats import norm
import time

_t0 = time.time()
_DEADLINE = 295.0  # safety margin before 300s

def acquisition(mu, sigma, f_best, cost):
    elapsed = time.time() - _t0
    if elapsed > _DEADLINE:
        return np.zeros_like(mu)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Clipped z for numerical stability
    z = np.clip((f_best - mu) / sigma_safe, -10, 10)
    
    # Standard EI for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Knowledge Gradient approximation:
    # VG (value of global information) ~ EI + E[max(0, f_best - mu - sigma*|z|)]
    # Approximate the second term as sigma * E[|z|] * P(mu < f_best)
    # E[|z|] for standard normal = sqrt(2/pi) ≈ 0.7979
    # P(mu < f_best) ≈ 0.5 when z is near 0, but we use a smoother:
    # Use a proxy: the expected reduction in the minimum is bounded by EI
    # plus an exploration term that peaks at moderate uncertainty.
    
    # Adaptive exploration weight based on remaining budget fraction
    # cost is the cost of evaluating this point; total budget is 30
    # When budget is plentiful, explore more; when scarce, exploit more
    remaining_frac = np.clip(1.0 - cost_safe / 30.0, 0.1, 1.0)
    beta = 0.3 * remaining_frac  # exploration coefficient
    
    # Exploration term: sigma-based, scaled by remaining budget
    exploration = beta * sigma_safe
    
    # Combine: cost-normalized (EI + exploration)
    score = (ei + exploration) / cost_safe
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
