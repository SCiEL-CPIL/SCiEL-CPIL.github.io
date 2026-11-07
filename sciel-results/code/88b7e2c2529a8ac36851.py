"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Distance-to-Best weighted EI with adaptive UCB.
    
    Uses Expected Improvement as the core exploitation signal, but scales it
    by an inverse-distance term to the best candidate (lowest mu) to create
    a repulsive force that pushes exploration toward distant, unexplored
    basins. This is critical for escaping the dense local minima of the
    Michalewicz function. An adaptive UCB component provides additional
    exploration pressure when uncertainty is high.
    """
    t_start = time.time()
    
    # Ensure proper types and handle numerical issues
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: replace non-finite values
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=-1e6)
    sigma = np.maximum(sigma, 1e-12)
    
    n = len(mu)
    eps = 1e-12
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Safety for EI
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Distance-to-Best weighting:
    # Find the index of the best candidate (lowest mu)
    best_idx = np.argmin(mu)
    best_mu = mu[best_idx]
    
    # Compute a "distance" proxy in function value space.
    # Since we don't have input coordinates, we use the spread of mu values
    # as a proxy for spatial distance. Points far from the best in mu-space
    # are likely in different basins.
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min
    
    if mu_range > eps:
        # Normalize mu to [0, 1] range
        mu_norm = (mu - mu_min) / mu_range
        # Distance from best in normalized space
        best_mu_norm = (best_mu - mu_min) / mu_range
        dist_from_best = np.abs(mu_norm - best_mu_norm)
        
        # Inverse distance weighting: closer to best gets lower weight,
        # farther gets higher weight. Use a soft inverse to avoid division by zero.
        # weight = 1 / (1 + alpha * dist_from_best)
        # But we want the OPPOSITE: we want to BOOST distant points.
        # So: weight = 1 + alpha * dist_from_best
        # This scales EI so that distant points get amplified.
        alpha = 2.0
        dist_weight = 1.0 + alpha * dist_from_best
    else:
        dist_weight = np.ones_like(mu)
    
    # Scaled EI
    ei_scaled = ei * dist_weight
    
    # Adaptive UCB component for additional exploration
    sigma_median = np.median(sigma)
    
    if mu_range > eps:
        rel_uncertainty = sigma_median / (mu_range + eps)
    else:
        rel_uncertainty = 1.0
    
    # Beta adapts: higher uncertainty -> more exploration
    beta = np.clip(1.5 * (0.5 + rel_uncertainty), 0.8, 2.5)
    
    # UCB score for minimization
    ucb_score = -mu + beta * sigma
    
    # Normalize both components to [0, 1] for blending
    ei_scaled_max = np.max(ei_scaled)
    ei_scaled_min = np.min(ei_scaled)
    if ei_scaled_max - ei_scaled_min > eps:
        ei_norm = (ei_scaled - ei_scaled_min) / (ei_scaled_max - ei_scaled_min)
    else:
        ei_norm = np.zeros_like(ei_scaled)
    
    ucb_max = np.max(ucb_score)
    ucb_min = np.min(ucb_score)
    if ucb_max - ucb_min > eps:
        ucb_norm = (ucb_score - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb_score)
    
    # Blend: weight more toward EI when the model is confident (low sigma),
    # and more toward UCB when uncertain
    confidence = np.mean(sigma < 0.1 * sigma_median + eps) if sigma_median > eps else 0.5
    weight_ei = 0.4 + 0.4 * confidence  # between 0.4 and 0.8
    weight_ucb = 1.0 - weight_ei
    
    score = weight_ei * ei_norm + weight_ucb * ucb_norm
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Safety: check time
    t_elapsed = time.time() - t_start
    if t_elapsed > 280:
        return -mu
    
    return score
# EVOLVE-BLOCK-END
