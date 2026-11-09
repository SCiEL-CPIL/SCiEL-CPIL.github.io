"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we are running out of time, use a fast fallback
    if time.time() - _START_TIME > _DEADLINE:
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, 0.0)

    epsilon = 1e-9
    sigma_safe = np.clip(sigma, 1e-9, None)
    cost_safe = np.maximum(cost, 1e-9)

    # 1. Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    # Clip z for numerical stability
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative (it should be by definition, but numerical issues can arise)
    ei = np.maximum(ei, 0.0)

    # 2. Probability of Improvement (PI)
    pi = norm.cdf(z_clipped)

    # 3. Cost normalization
    # Cost is 1 + mean(x_scaled). We want to favor cheaper points.
    # Normalizing to [0, 1] allows for relative weighting.
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range < 1e-12:
        cost_norm = np.zeros_like(cost)
    else:
        cost_norm = (cost - cost_min) / cost_range

    # Cost weight: Cheaper points get higher weight.
    # Using an inverse relationship with a floor to avoid division by zero or extreme spikes.
    # cost_weight = 1 / (cost_norm + 0.5) seems reasonable. 
    # Or simpler: just divide by cost directly? 
    # Inspiration 1 used 1/(cost_norm + 0.1). Let's stick to a robust cost penalty.
    # Dividing by raw cost is a strong penalty. Let's use a normalized penalty.
    # Let's try: cost_weight = 1.0 / (1.0 + cost_norm)
    # This ranges from 1.0 (cheapest) to 0.5 (most expensive).
    cost_weight = 1.0 / (1.0 + cost_norm)

    # 4. Adaptive Exploration/Exploitation
    # When we are close to the optimum (f_best is low), we should exploit more.
    # When f_best is high, we should explore more.
    # However, we don't know the true optimum. 
    # Instead, let's look at the uncertainty.
    # High sigma -> Explore. Low sigma -> Exploit.
    
    # Normalize sigma to [0, 1]
    sigma_max = np.max(sigma_safe)
    if sigma_max < 1e-12:
        sigma_norm = np.zeros_like(sigma_safe)
    else:
        sigma_norm = sigma_safe / sigma_max

    # Blend EI and PI
    # PI is a 0-1 probability. EI is unbounded.
    # We need to scale PI to be comparable to EI.
    # Let's scale PI by the typical improvement scale.
    # A simple heuristic: scale PI by the max EI in the batch?
    # Or just use a fixed weight?
    # Inspiration 1 used normalized EI and PI.
    
    ei_max = np.max(ei)
    if ei_max < 1e-12:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = ei / ei_max
        
    pi_norm = pi # Already 0-1

    # Dynamic weight:
    # If we have high uncertainty everywhere, explore (PI).
    # If we have low uncertainty, exploit (EI).
    # Let's use the mean sigma as a proxy for global uncertainty level.
    # But we want point-wise adaptation.
    
    # Let's try a fixed blend first, as it's more robust.
    # EI is the primary driver. PI helps in flat regions where EI is small but probability is decent.
    # Let's weight EI heavily.
    alpha = 0.8  # Weight for EI
    beta = 0.2   # Weight for PI
    
    # Combine
    combined = alpha * ei_norm + beta * pi_norm
    
    # Apply cost weight
    score = combined * cost_weight
    
    # Ensure output is finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # If all scores are 0 (e.g. all ei=0), fall back to pure exploration (sigma) or random?
    # If ei is 0 everywhere, it means mu >= f_best everywhere.
    # In that case, we should pick the point with highest sigma (most uncertain) and lowest cost.
    if np.max(score) < 1e-9:
         # Fallback: maximize sigma / cost
         fallback_score = sigma_safe / cost_safe
         score = np.nan_to_num(fallback_score, nan=0.0, posinf=0.0, neginf=0.0)

    return score
# EVOLVE-BLOCK-END
