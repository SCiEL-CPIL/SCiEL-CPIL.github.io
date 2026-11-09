"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 250.0
    
    # Numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Pareto-frontier guided exploration:
    # 1. Find the candidate with the lowest predicted mean (exploitation anchor)
    min_mu_idx = np.argmin(mu)
    anchor_mu = mu[min_mu_idx]
    
    # 2. Compute normalized distance from anchor in prediction space
    #    This measures how far each candidate is from the most promising region
    #    in terms of the GP posterior mean
    mu_range = np.ptp(mu)
    if mu_range < 1e-10:
        mu_range = 1e-10
    dist_from_anchor = np.abs(mu - anchor_mu) / mu_range
    
    # 3. Compute uncertainty-weighted exploration score
    #    High uncertainty + far from anchor = good exploration candidate
    #    Low uncertainty + near anchor = good exploitation candidate
    sigma_max = np.max(sigma_safe)
    if sigma_max < 1e-10:
        sigma_max = 1e-10
    normalized_sigma = sigma_safe / sigma_max
    
    # 4. Combine exploitation and exploration
    #    Exploitation: prefer candidates with low predicted mean (close to f_best or lower)
    #    Exploration: prefer candidates with high uncertainty, weighted by distance from anchor
    #
    #    Score = alpha * exploitation + beta * exploration
    #    where exploitation = 1 - (mu - min_mu) / (max_mu - min_mu)  [higher = better]
    #    exploration = normalized_sigma * (1 + dist_from_anchor)
    
    # Exploitation score: higher when mu is close to the minimum
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_span = mu_max - mu_min
    if mu_span < 1e-10:
        mu_span = 1e-10
    exploitation = 1.0 - (mu - mu_min) / mu_span
    
    # Exploration score: high uncertainty, boosted by distance from anchor
    exploration = normalized_sigma * (1.0 + dist_from_anchor)
    
    # Weighted combination
    alpha = 0.7  # exploitation weight
    beta = 0.3   # exploration weight
    
    scores = alpha * exploitation + beta * exploration
    
    # Add a small EI component to maintain gradient toward improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    ei_max = np.max(ei)
    if ei_max > 1e-10:
        ei_normalized = ei / ei_max
    else:
        ei_normalized = np.zeros_like(ei)
    
    scores = 0.8 * (alpha * exploitation + beta * exploration) + 0.2 * ei_normalized
    
    # Check time
    if time.time() > deadline:
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
