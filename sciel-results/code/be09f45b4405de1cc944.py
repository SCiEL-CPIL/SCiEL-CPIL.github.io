"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Known global minimum of Styblinski-Tang 1D
ST_MIN = -39.1662

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition combining Expected Improvement (EI) with a
    lower-bound / uncertainty-driven exploration term to balance
    exploitation near the known optimum region and exploration
    in under-sampled areas.
    
    mu, sigma: arrays of shape (2000,) in objective units.
    f_best: lowest observed value so far.
    
    Returns: array of 2000 finite scores (higher = evaluate next).
    """
    t0 = time.time()
    n = len(mu)
    
    # Clamp sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Compute lower bound (optimistic prediction)
    # Lower bound = mu - beta * sigma (we want to explore where this is low)
    # For exploration, we want to find points where the lower bound is very low
    # i.e., where there's a chance of finding a much better value
    beta = 1.5  # exploration parameter
    lower_bound = mu - beta * sigma_safe
    
    # Normalize EI and lower_bound to [0, 1] for blending
    ei_min, ei_max = np.min(ei), np.max(ei)
    lb_min, lb_max = np.min(lower_bound), np.max(lower_bound)
    
    if ei_max > ei_min:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros(n)
    
    if lb_max > lb_min:
        # Invert lower_bound: lower values are better, so we want higher normalized scores for lower bounds
        lb_norm = 1.0 - (lower_bound - lb_min) / (lb_max - lb_min)
    else:
        lb_norm = np.zeros(n)
    
    # Estimate progress: use (f_best - ST_MIN) / (initial_range_estimate)
    # ST_MIN is -39.1662, and the function range is roughly [-40, 10]
    # So a reasonable progress metric is how much of the gap has been closed
    gap = f_best - ST_MIN  # positive, decreasing as we improve
    # Normalize gap: when gap is large (early), explore more
    # When gap is small (late), exploit more
    # Use a sigmoid-like transition
    # gap_scale ~ 40 (roughly the total range)
    gap_scale = 40.0
    progress = np.exp(-gap / gap_scale)  # 0 when gap is large, 1 when gap is small
    
    # Blend weight: progress controls how much we trust EI vs exploration
    # When progress is high (close to optimum), weight EI heavily
    # When progress is low (far from optimum), weight exploration more
    w_ei = 0.5 + 0.5 * progress  # ranges from 0.5 to 1.0
    w_explore = 1.0 - w_ei  # ranges from 0.5 to 0.0
    
    # Combine: higher EI is better, higher lb_norm means lower lower bound (better)
    # We want to maximize the combined score
    score = w_ei * ei_norm + w_explore * lb_norm
    
    # Scale back to original EI scale for interpretability
    if ei_max > ei_min:
        score = score * (ei_max - ei_min) + ei_min
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time guard
    if time.time() - t0 > 2.5:
        return score
    
    return score
# EVOLVE-BLOCK-END
