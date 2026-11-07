"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    prizes: shape (5, 100), uniform in [0,1]
    weights: shape (5, 100), uniform in [0,1]
    Returns: array of shape (100,) with attractiveness scores (larger = more attractive)
    """
    # Ensure proper shapes
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        raise ValueError("Expected shape (5, 100) for prizes and weights")
    
    # Per-constraint efficiency: prize / weight for each constraint
    # Avoid division by zero
    w_safe = np.maximum(weights, 1e-12)
    ratios = prizes / w_safe  # shape (5, 100)
    
    # Sum-ratio: total prize / total weight (aggregate efficiency)
    sum_ratio = prizes.sum(axis=0) / (weights.sum(axis=0) + 1e-12)  # shape (100,)
    
    # Geometric mean of ratios (smooth min via log-sum-exp with negative)
    # log(mean of exp) -> use logsumexp for stability
    log_ratios = np.log(np.maximum(ratios, 1e-15))  # shape (5, 100)
    
    # Geometric mean = exp(mean(log(ratios)))
    geo_mean = np.exp(log_ratios.mean(axis=0))  # shape (100,)
    
    # Smooth min approximation using log-sum-exp with large negative scaling
    # min(x1,...,x5) ≈ -logsumexp(-x_i * beta) / beta for large beta
    beta = 10.0
    neg_scaled = -ratios * beta
    # Subtract max for numerical stability
    max_neg_scaled = neg_scaled.max(axis=0, keepdims=True)
    logsumexp_neg = max_neg_scaled.squeeze(axis=0) + np.log(np.exp(neg_scaled - max_neg_scaled).sum(axis=0))
    smooth_min = -logsumexp_neg / beta  # shape (100,)
    
    # Blend: combine sum_ratio, geo_mean, and smooth_min
    # Normalize each to [0,1] range for fair blending
    def normalize(arr):
        mn = arr.min()
        mx = arr.max()
        if mx - mn < 1e-15:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)
    
    nr_sum = normalize(sum_ratio)
    nr_geo = normalize(geo_mean)
    nr_smin = normalize(smooth_min)
    
    # Weighted blend: emphasize bottleneck awareness while keeping aggregate efficiency
    blended = 0.3 * nr_sum + 0.35 * nr_geo + 0.35 * nr_smin
    
    # Convert back to positive scores
    # Add small offset to ensure all positive
    scores = 1e-6 + blended
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
