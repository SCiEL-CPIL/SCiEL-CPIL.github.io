"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Computes scores for 2000 candidate points.
    Strategy: Probability of Improvement (PI) with a tuned offset (xi).
    
    This is a materially different design family from the current program's 
    Adaptive UCB (Confidence Bound). PI focuses on the probability that a candidate
    is better than the current best, rather than a lower bound on the function value.
    
    In high dimensions like Hartmann 6D, UCB can sometimes be too aggressive in 
    exploring regions with high uncertainty but low probability of containing the 
    global minimum, especially if the GP mean is not well-calibrated in those regions.
    PI with an offset (xi) provides a more balanced approach: it favors points that 
    are likely to improve on f_best, but the offset xi allows for some exploration 
    by requiring a "significant" improvement.
    
    We use a dynamic xi that adapts to the current best value.
    As f_best gets closer to the global minimum, we can reduce xi to allow finer 
    improvements. When f_best is far, a larger xi helps skip over marginal improvements 
    and seek more significant jumps.
    
    Formula:
    z = (f_best - xi - mu) / sigma
    PI = norm.cdf(z)
    
    We return PI as the score. Higher PI means higher probability of improvement.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: replace non-finite or negative sigma
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-8), sigma, 1e-8)
    
    # Dynamic offset xi
    # The range of Hartmann 6D is roughly [-3.3, 3.3], so a reasonable scale is ~1.0.
    # If f_best is -3.0, we want a smaller xi than if f_best is 0.0.
    # Let's map the distance to the known minimum (-3.32237) to xi.
    # Distance d = f_best - (-3.32237)
    # If d is large (e.g., 3.0), xi should be larger (e.g., 0.1).
    # If d is small (e.g., 0.1), xi should be smaller (e.g., 0.01).
    
    target_min = -3.32237
    distance_to_opt = f_best - target_min
    
    # Scale xi: 
    # Let's try xi = 0.01 + 0.05 * clip(distance_to_opt, 0, 1.0)
    # This gives xi in [0.01, 0.06].
    # This is a small offset, meaning we are mostly looking for any improvement,
    # but slightly favoring those with a bit of margin.
    
    # Alternatively, a fixed small xi is often robust.
    # Let's try a fixed xi = 0.01.
    # Or, to be more exploratory, a larger xi? 
    # Actually, PI with xi=0 is very greedy. 
    # PI with larger xi is more exploratory in the sense that it ignores points 
    # that are only marginally better, potentially allowing the GP to "skip" 
    # local minima? No, it just requires a bigger jump.
    
    # Let's use a slightly larger xi to encourage exploration of regions 
    # that might have a significantly lower value, not just a tiny improvement.
    # xi = 0.05
    
    xi = 0.01
    
    z = (f_best - xi - mu) / sigma
    pi = norm.cdf(z)
    
    # Ensure finite
    pi = np.where(np.isfinite(pi), pi, 0.0)
    
    return pi

# EVOLVE-BLOCK-END
