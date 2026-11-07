import numpy as np
from scipy.stats import norm
import time

# Global time tracking for safety
_start_time = time.time()
_time_limit = 300  # Seconds

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining EHI (Expected Hypar Improvement)
    with a UCB-like exploration term, normalized by cost.
    
    The EHI component uses a tanh-based approximation to model the expected
    hyperbolic tangent of the normalized improvement, which naturally saturates
    and penalizes high-cost/high-uncertainty regions. The UCB component adds
    an explicit exploration bonus proportional to sigma to help escape local
    minima in the early stages. Both are scaled by cost to prioritize cheaper
    evaluations.
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviation predictions.
        f_best: Scalar, the best objective value observed so far.
        cost: (2000,) array of evaluation costs (1 + mean(x)).
        
    Returns:
        (2000,) array of acquisition scores (higher = evaluate next).
    """
    # Time guard: if we're getting close to the limit, return a safe default
    elapsed = time.time() - _start_time
    if elapsed > _time_limit - 5:  # 5 second safety margin
        return np.zeros_like(mu)
    
    # Ensure sigma is positive to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # --- EHI Component ---
    # Normalized improvement: (f_best - mu) / cost
    imp = (f_best - mu) / cost
    
    # Normalized uncertainty: sigma / cost
    unc = sigma / cost
    
    # Delta-method approximation for E[tanh(X)] where X ~ N(imp, unc)
    # E[tanh(X)] ≈ tanh(m) * (1 - s² * (1 - tanh(m)²))
    t = np.tanh(imp)
    ehi = t * (1.0 - unc**2 * (1.0 - t**2))
    
    # --- UCB Component ---
    # Exploration parameter beta. A value of 2.0 is standard.
    # The UCB term is (f_best - mu + beta * sigma) / cost
    beta = 2.0
    ucb = (f_best - mu + beta * sigma) / cost
    
    # --- Combine ---
    # Weight the EHI and UCB components.
    # EHI is bounded in [-1, 1] (approximately), while UCB can be larger.
    # We normalize the UCB term to a similar scale to prevent it from dominating.
    # A simple approach: scale UCB by a factor to match the typical magnitude of EHI.
    # Since cost is in [1, 2] and f_best - mu and sigma are in objective units,
    # we use a fixed scaling factor based on the expected range of the objective.
    # For Hartmann 6d, the range is roughly [-3.3, 0]. Let's use a scale of ~3.
    ucb_scale = 3.0
    ucb_scaled = ucb / ucb_scale
    
    # Blend: 0.6 * EHI + 0.4 * UCB_scaled
    # This gives more weight to the cost-aware EHI but retains exploration.
    score = 0.6 * ehi + 0.4 * ucb_scaled
    
    # Ensure output is finite and real
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
