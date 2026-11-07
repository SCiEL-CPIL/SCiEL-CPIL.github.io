import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for standard normal CDF/PDF
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Hard cost gate strategy:
    # Cost is in [1, 2]. The minimum cost is 1 (bottom-left corner).
    # We want to strongly prefer cheap points.
    # Instead of dividing by cost, we create a sharp threshold.
    # Let's look at the distribution of costs.
    # If a point is "expensive" (e.g., cost > 1.2), we heavily penalize it
    # unless its EI is significantly higher than the best cheap EI.
    
    # Compute the maximum EI among cheap points (cost < 1.1)
    # This acts as a baseline for "good enough" improvement in cheap regions.
    cheap_mask = cost < 1.1
    if np.any(cheap_mask):
        ei_cheap_max = np.max(ei[cheap_mask])
    else:
        ei_cheap_max = 0.0
    
    # For expensive points, we require their EI to be significantly higher than the best cheap EI
    # to be considered. Otherwise, we dampen them heavily.
    # This forces the algorithm to keep picking cheap points until the GP is very confident
    # that no better cheap point exists, or until an expensive point offers a huge potential gain.
    
    # Dampening factor for expensive points
    # If cost is high, and EI is not much better than the best cheap EI, score drops.
    
    # Let's use a soft threshold with a steep slope.
    # Threshold for cost: start penalizing above 1.05.
    # Penalty increases as cost increases.
    
    # Alternative structural change:
    # Use a "Cost-Adjusted EI" that is zero for expensive points unless EI is very high.
    
    # Let's try a multiplicative gate.
    # Gate = 1 if cost is low, decays rapidly if cost is high.
    # But we also want to allow expensive points if they are promising.
    
    # Let's define a "relative cost" where 1.0 is min, 2.0 is max.
    # We want to prioritize the bottom-left (low cost).
    
    # Strategy: 
    # Score = EI * CostFactor
    # CostFactor is 1 for cost <= 1.05, and drops exponentially for cost > 1.05.
    # However, if EI is huge, we might still want to pick it.
    # So: Score = EI * exp(-lambda * max(0, cost - 1.05))
    
    # Let's pick lambda to make the decay steep.
    # If cost = 1.5, max(0, 0.45) = 0.45. exp(-10 * 0.45) = exp(-4.5) ~ 0.01.
    # This is a strong penalty.
    
    # Let's try lambda = 5.
    # cost = 1.5 -> exp(-2.25) ~ 0.1.
    # cost = 2.0 -> exp(-4.75) ~ 0.008.
    
    # This seems reasonable. It heavily favors cheap points.
    
    cost_penalty = np.exp(-5.0 * np.maximum(0.0, cost - 1.05))
    
    score = ei * cost_penalty
    
    # Add a tiny exploration term to avoid zero scores if EI is zero everywhere
    # This helps if the GP is overconfident.
    exploration = 1e-6 * sigma_safe * cost_penalty
    
    score = score + exploration
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
