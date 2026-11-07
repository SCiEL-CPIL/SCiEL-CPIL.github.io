import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Structural change: Use a Pareto-frontier based cost-aware acquisition.
    
    Instead of linearly blending EI and PI with adaptive weights, we:
    1. Compute the expected improvement (EI) for minimization
    2. Compute a "regret bound" that accounts for uncertainty
    3. Use a threshold-based selection: only consider candidates where
       the expected improvement is sufficiently large relative to cost
    4. Among valid candidates, rank by improvement-per-cost with a
       multiplicative penalty for high uncertainty (to avoid noisy regions)
    
    This creates a non-linear, threshold-gated scoring that naturally
    separates exploitation (high EI, low cost) from exploration (high
    uncertainty, but only if it could lead to large improvement).
    """
    t0 = time.time()
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score for improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (for minimization)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Improvement per cost
    ei_per_cost = ei / cost_safe
    
    # Uncertainty penalty: penalize regions with high sigma relative to
    # the potential improvement. This prevents the acquisition from
    # being dominated by pure exploration in high-uncertainty regions.
    # The penalty is multiplicative and scales with the ratio of sigma
    # to the expected improvement potential.
    # When EI is large relative to sigma, penalty is small (exploit)
    # When sigma is large relative to EI, penalty is large (explore less)
    sigma_ratio = sigma_safe / (ei + sigma_safe)
    # Penalty: 1.0 when sigma_ratio is small (low uncertainty relative to improvement)
    # Penalty: 0.3 when sigma_ratio is large (high uncertainty)
    uncertainty_penalty = 1.0 - 0.7 * sigma_ratio
    
    # Base score: improvement per cost, penalized by uncertainty
    score = ei_per_cost * uncertainty_penalty
    
    # Threshold gating: only consider candidates where EI is above a
    # small fraction of f_best. This creates a hard constraint that
    # prevents selecting points with negligible expected improvement.
    # The threshold adapts to the current best value.
    threshold = 0.01 * abs(f_best) if abs(f_best) > 1e-6 else 0.001
    mask = ei > threshold
    
    # For candidates below threshold, assign a very low score
    score = np.where(mask, score, 0.0)
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Safety deadline guard
    elapsed = time.time() - t0
    if elapsed > 290:
        return np.maximum(ei_per_cost, 0.0)
    
    return score
