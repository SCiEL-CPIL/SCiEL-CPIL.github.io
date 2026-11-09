import numpy as np
import time
from scipy.special import erf

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle zero sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1.0)  # cost is in [1, 2]
    
    # Check deadline
    if time.time() > deadline:
        z = (f_best - mu) / sigma_safe
        z_clipped = np.clip(z, -50, 50)
        phi_z = np.exp(-0.5 * z_clipped ** 2) / np.sqrt(2 * np.pi)
        cdf_z = 0.5 * (1.0 + erf(z_clipped / np.sqrt(2.0)))
        ei = (f_best - mu) * cdf_z + sigma * phi_z
        ei = np.maximum(ei, 0.0)
        scores = ei / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -50, 50)
    
    # Compute phi(z) and Phi(z)
    phi_z = np.exp(-0.5 * z_clipped ** 2) / np.sqrt(2 * np.pi)
    cdf_z = 0.5 * (1.0 + erf(z_clipped / np.sqrt(2.0)))
    
    # Expected Improvement
    ei = (f_best - mu) * cdf_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        scores = ei / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Compute Probability of Improvement (PoI)
    poi = cdf_z  # P(f(x) < f_best)
    
    # Compute a confidence-adjusted improvement metric
    # Blend EI with PoI to get a more robust signal
    # The idea: when PoI is high, we're confident in improvement
    # when PoI is low, EI may be driven by high uncertainty
    
    # Normalize EI and PoI to comparable scales
    ei_max = np.max(ei)
    poi_max = np.max(poi)
    
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    if poi_max > 1e-12:
        poi_norm = poi / poi_max
    else:
        poi_norm = np.zeros_like(poi)
    
    # Adaptive blending: weight EI more when there's actual improvement potential
    # weight PoI more when uncertainty is high
    sigma_mean = np.mean(sigma)
    sigma_max = np.max(sigma)
    
    if sigma_max > 1e-9:
        sigma_rel = sigma / sigma_max
    else:
        sigma_rel = np.zeros_like(sigma)
    
    # When sigma is high relative to max, trust PoI more (exploration)
    # When sigma is low, trust EI more (exploitation)
    blend_weight = 0.5 + 0.5 * sigma_rel  # [0.5, 1.0]
    
    # Combined improvement score
    combined = blend_weight * ei_norm + (1.0 - blend_weight) * poi_norm
    combined = np.maximum(combined, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        scores = combined / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Cost-aware scoring with adaptive penalty
    # Estimate the stage of optimization
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min
    
    if mu_range > 1e-9:
        f_best_rel = (f_best - mu_min) / mu_range
        f_best_rel = np.clip(f_best_rel, 0, 1)
    else:
        f_best_rel = 0.5
    
    # Cost exponent: lower when exploring, higher when exploiting
    cost_exp = 0.3 + 0.7 * f_best_rel
    
    # Per-point adjustment: high uncertainty -> lower cost penalty
    cost_exp_adjusted = cost_exp * (0.7 + 0.3 * (1.0 - sigma_rel))
    
    # Base score: combined improvement divided by cost penalty
    scores = combined / (cost_safe ** cost_exp_adjusted)
    
    # Check deadline
    if time.time() > deadline:
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Add a small exploration bonus for high-uncertainty regions
    exploration_bonus = 0.1 * sigma_rel * combined
    
    scores += exploration_bonus
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Final deadline check
    if time.time() > deadline:
        return np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
