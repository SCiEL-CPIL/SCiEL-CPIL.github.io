import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Safety: ensure finite inputs and avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -10, 10)
    
    # Compute Expected Improvement (for minimization)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware adjustment:
    # Combine additive and multiplicative cost penalties for robustness.
    # The additive term (lambda * cost) provides a direct gradient toward cheaper regions.
    # The multiplicative term (1 / (1 + alpha * (cost - 1))) softly downweights expensive points.
    
    lambda_additive = 0.1
    alpha_multiplicative = 0.1
    
    # Additive penalty: linear in cost
    cost_penalty_additive = lambda_additive * cost
    
    # Multiplicative factor: mild penalty for higher cost
    # cost is in [1, 2], so (cost - 1) is in [0, 1]
    # factor ranges from 1.0 (cost=1) to 1/(1+0.1) ≈ 0.909 (cost=2)
    cost_factor = 1.0 / (1.0 + alpha_multiplicative * (cost - 1.0))
    
    # Combine: apply multiplicative factor first, then subtract additive penalty
    # This ensures that even points with high EI are penalized if they are expensive
    scores = ei * cost_factor - cost_penalty_additive
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
