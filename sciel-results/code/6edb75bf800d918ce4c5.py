# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 300.0  # seconds, must finish before 330s wall-clock


def _check_deadline():
    return (time.time() - _START_TIME) > (_DEADLINE - 5.0)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Levy 2D.
    
    Strategy: Combines cost-normalized Expected Improvement with a diversity
    penalty derived from GP uncertainty (sigma). Candidates in regions of high
    uncertainty (unexplored) get a bonus, while those in low-uncertainty regions
    (well-explored) are penalized relative to their EI. This is a structural
    shift from pure pointwise EI/cost to a diversity-aware scoring.
    
    We use a hybrid:
    1. Base score: cost-normalized EI (exploitation)
    2. Diversity bonus: scaled by sigma (exploration of uncertain regions)
    3. Combined with a weighted sum, where the weight adapts based on how
       concentrated the high-EI candidates are (measured via sigma distribution).
    """
    if _check_deadline():
        # Return a safe, deterministic fallback if we're near the deadline
        # Pick the candidate with lowest cost and reasonable mu
        idx = np.argmin(cost)
        scores = np.full_like(mu, -1e10)
        scores[idx] = 1.0
        return scores

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # --- Component 1: Cost-normalized Expected Improvement ---
    # Standard EI: (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI is non-negative
    ei_cost = ei / cost_safe
    
    # --- Component 2: Diversity / Exploration bonus ---
    # High sigma means unexplored. We want to reward candidates that are
    # in uncertain regions, especially if they also have decent EI.
    # Use sigma as a proxy for "distance from evaluated points" (low sigma = close to data).
    # Normalize sigma to [0, 1] for stable weighting.
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    sigma_range = sigma_max - sigma_min
    if sigma_range > 1e-12:
        sigma_norm = (sigma_safe - sigma_min) / sigma_range
    else:
        sigma_norm = np.ones_like(sigma_safe) * 0.5
    
    # Exploration score: higher for uncertain regions
    exploration = sigma_norm
    
    # --- Component 3: Adaptive weighting ---
    # If the top EI candidates are all in low-sigma regions (concentrated),
    # increase the weight on exploration to force diversity.
    # Measure "concentration" of high-EI candidates in low-sigma regions.
    
    # Sort by EI (descending)
    ei_sorted = np.sort(ei)[::-1]
    sigma_sorted = sigma_safe[np.argsort(ei)[::-1]]
    
    # Take top 10% of candidates by EI
    top_k = max(1, n // 10)
    top_ei = ei_sorted[:top_k]
    top_sigma = sigma_sorted[:top_k]
    
    # If the top-EI candidates have low average sigma, they're concentrated in explored regions
    mean_top_sigma = np.mean(top_sigma)
    global_mean_sigma = np.mean(sigma_safe)
    
    # Concentration ratio: if top sigma << global sigma, we're stuck
    if global_mean_sigma > 1e-12:
        concentration = mean_top_sigma / global_mean_sigma
    else:
        concentration = 1.0
    
    # If concentration is low (top candidates in low-sigma regions), boost exploration
    # Blend: weight_exploration increases as concentration decreases
    # Use a smooth transition
    # When concentration ~ 1.0 (uniform), weight_exploration ~ 0.3
    # When concentration ~ 0.1 (concentrated), weight_exploration ~ 0.7
    weight_exploration = 0.3 + 0.4 * (1.0 - np.clip(concentration, 0.0, 1.0))
    weight_exploitation = 1.0 - weight_exploration
    
    # --- Combine scores ---
    # Normalize ei_cost to [0, 1] for stable combination
    ei_cost_min = np.min(ei_cost)
    ei_cost_max = np.max(ei_cost)
    ei_cost_range = ei_cost_max - ei_cost_min
    if ei_cost_range > 1e-12:
        ei_cost_norm = (ei_cost - ei_cost_min) / ei_cost_range
    else:
        ei_cost_norm = np.ones_like(ei_cost) * 0.5
    
    # Final score: weighted combination
    scores = weight_exploitation * ei_cost_norm + weight_exploration * exploration
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
