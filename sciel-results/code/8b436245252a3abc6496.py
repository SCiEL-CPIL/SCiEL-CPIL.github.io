import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle zero sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement (PoI)
    poi = cdf_z
    
    # Use a hybrid acquisition that balances exploitation (EI) and exploration (PoI/Ucb)
    # We'll use a combination of EI and a UCB-like term to encourage exploration
    # when uncertainty is high relative to the current best
    
    # UCB-like score: mu - kappa * sigma (for minimization, we want low mu and high sigma)
    # But we need to normalize. Let's use a simpler approach:
    # Blend EI with a normalized PoI term that encourages exploration
    
    # Adaptive blending: when sigma is large relative to the spread of mu,
    # we want more exploration. When sigma is small, we want more exploitation.
    
    # Compute a relative uncertainty measure
    sigma_mean = np.mean(sigma_safe)
    sigma_std = np.std(sigma_safe)
    
    # If all sigmas are similar, use a fixed blend
    # If sigmas vary, weight points with higher sigma more for exploration
    
    # Use a soft blend: 0.7 * EI + 0.3 * (normalized PoI)
    # But scale PoI by the relative uncertainty to make it meaningful
    
    # Normalize PoI to [0, 1] range (it already is)
    # Scale by a factor that depends on the overall uncertainty level
    if sigma_mean > 1e-9:
        # Higher uncertainty -> more weight on exploration
        exploration_weight = min(0.5, sigma_mean / (sigma_mean + abs(f_best) + 1e-9))
    else:
        exploration_weight = 0.1
    
    # Combined score before cost adjustment
    combined = (1.0 - exploration_weight) * ei + exploration_weight * poi * sigma_safe
    
    # Cost-aware normalization
    # Cost is in [1, 2] (1 + mean(x_scaled) where x_scaled in [0,1])
    # Use a power penalty to discourage high-cost evaluations
    # A softer penalty (power ~1.0-1.5) allows some exploration of higher-cost regions
    cost_safe = np.maximum(cost, 1e-9)
    scores = combined / (cost_safe ** 1.2)
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 325:
        return np.zeros_like(mu)
    
    return scores
