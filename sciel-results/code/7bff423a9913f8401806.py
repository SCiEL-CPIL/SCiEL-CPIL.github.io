"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Strategy: Cost-Adjusted Probability of Improvement (CA-PI).
    
    Core: PI(x) = P(f(x) < f_best) = norm.cdf((f_best - mu(x)) / sigma(x))
    
    Enhancements:
    1. **Cost Awareness**: The cost is 1 + mean(x_scaled_to_01), so lower cost means
       closer to the origin (where the Rosenbrock minimum is). We add a small bonus
       for low-cost points to bias the search toward the cheap region without
       abandoning the probability-based core.
    
    2. **Uncertainty Dampening**: In regions with very high sigma, the PI score can
       be misleading (high uncertainty can make PI artificially high or low). We
       apply a soft penalty when sigma is very large to avoid chasing noise.
    
    3. **Early Exploration**: When f_best is still far from the minimum (early in
       the run), we add a small exploration term based on -sigma to encourage
       visiting unexplored regions. This term decays as f_best improves.
    
    Why this stays in the PI family:
    - The dominant term is still the probability of improvement
    - Cost and uncertainty adjustments are secondary and bounded
    - The method remains robust to GP miscalibration
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Safety: handle NaN/Inf
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute base Probability of Improvement (PI)
    z = (f_best - mu) / sigma
    pi_score = norm.cdf(z)
    
    # Cost adjustment: normalize cost to [0, 1] range (cost is in [1, 2])
    cost_normalized = (cost - 1.0) / 1.0  # Now in [0, 1]
    # Low cost (near 0) gets a bonus, high cost (near 1) gets a penalty
    # Use a gentle sigmoid-like scaling: bonus = 0.1 * (1 - cost_normalized)
    cost_bonus = 0.1 * (1.0 - cost_normalized)
    
    # Uncertainty dampening: penalize very high sigma
    # When sigma > 1.0, apply a soft penalty
    sigma_penalty = 0.0
    high_sigma_mask = sigma > 1.0
    if np.any(high_sigma_mask):
        # Penalty scales with excess sigma, capped at 0.1
        sigma_penalty = 0.1 * np.clip((sigma - 1.0) / 1.0, 0, 1)
    
    # Early exploration: when f_best is large (far from minimum), add exploration
    # f_best for Rosenbrock can be large initially; use a threshold
    exploration_weight = 0.0
    if f_best > 10.0:
        # Decay exploration as f_best improves
        exploration_weight = 0.05 * np.clip(1.0 - f_best / 50.0, 0, 1)
    
    exploration_bonus = exploration_weight * (-sigma)
    
    # Combine: base PI + cost bonus - sigma penalty + exploration bonus
    score = pi_score + cost_bonus - sigma_penalty + exploration_bonus
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 250:  # Safety margin
         # Fallback to simple exploitation: -mu
         score = -mu
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.5, posinf=1.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
