import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Reference cost: maximum possible cost is 2.0 (1 + mean([1, 1]))
    # Using a fixed reference for hypervolume calculation
    c_ref = 2.0
    
    # Calculate Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    # Handle sigma == 0
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # For sigma -> 0, EI -> max(0, f_best - mu)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Correct for the case where sigma was very small
    # If sigma was < 1e-9, the formula above might have artifacts if z is large
    # But norm.cdf(large) ~ 1, norm.pdf(large) ~ 0.
    # (f_best - mu) * 1 + 0 = f_best - mu.
    # We need max(0, f_best - mu).
    # If mu > f_best, z is negative. cdf is small.
    # Let's explicitly mask for sigma < 1e-9
    mask_small_sigma = sigma < 1e-9
    ei[mask_small_sigma] = np.maximum(0, f_best - mu[mask_small_sigma])
    
    # Ensure EI is non-negative
    ei = np.maximum(0, ei)
    
    # Hypervolume contribution factor: (c_ref - cost)
    # Cost is in [1, 2]. c_ref = 2.
    # If cost > c_ref, factor is negative. But cost <= 2.
    # So factor is in [0, 1].
    cost_saving = c_ref - cost
    cost_saving = np.maximum(0, cost_saving)
    
    # Score = EI * CostSaving
    scores = ei * cost_saving
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
