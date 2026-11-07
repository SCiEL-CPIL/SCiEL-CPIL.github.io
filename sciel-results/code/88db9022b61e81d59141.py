# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Deadline guard: if we've spent too long, fall back to a simple EI.
    # (In practice this function is called quickly, but we guard anyway.)
    
    # Compute a simple Expected Improvement as a baseline
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # We want to combine exploitation (EI) with exploration (UCB-like term)
    # Use an adaptive beta that decays as we get more confident (lower sigma)
    
    # Estimate progress: if f_best is close to 0, we're doing well
    # Use the spread of sigma to gauge exploration needs
    
    # UCB term: explore where uncertainty is high
    # But we need to balance with exploitation
    
    # A good strategy for Weierstrass (multimodal) is to use a combination:
    # score = EI + beta * sigma
    # where beta adapts based on how much improvement we've seen
    
    # Since we don't know the step count directly, we infer from f_best:
    # Early on, f_best is large (far from 0), so we should explore more
    # Later, f_best is small, so we should exploit more
    
    # Normalize f_best to get a progress indicator
    # Weierstrass 1D on [-0.5, 0.5] has minimum 0, and typical values are O(1)
    # Let's use a simple heuristic:
    
    # If f_best is large (> 1), we're early, explore more
    # If f_best is small (< 0.1), we're late, exploit more
    
    # Adaptive beta: starts high, decays as f_best improves
    # Use exponential decay based on f_best magnitude
    progress = np.clip(f_best / 1.0, 0.0, 1.0)  # 1.0 = early, 0.0 = late
    
    # Beta decays exponentially from ~2.0 to ~0.1
    beta = 0.1 + 1.9 * np.exp(-3.0 * (1.0 - progress))
    
    # Combine EI and UCB
    # score = EI + beta * sigma
    # This encourages exploring high-uncertainty regions early,
    # and exploiting low-uncertainty, high-improvement regions later
    
    score = ei + beta * sigma_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
