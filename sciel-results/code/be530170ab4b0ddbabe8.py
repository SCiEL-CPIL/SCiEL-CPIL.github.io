"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 285.0  # Safety margin before 300s limit

def _check_deadline():
    return time.time() >= _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: return a safe default if we're running out of time
    if _check_deadline():
        # Return cost-based scores (prefer low cost) as a safe fallback
        return 1.0 / (cost + 1e-9)
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring: use a moderate penalty
    # The cost is 1 + mean(x_scaled), so central points have lower cost
    # We want to balance improvement with cost efficiency
    
    # Primary: EI with linear cost penalty
    score_ei = ei / (cost + 0.5)
    
    # Secondary: uncertainty-weighted exploration
    # Points with high sigma are worth exploring
    # But also prefer points in the central region (lower cost, higher prior probability)
    # The moved minimum is in inner 80%: scaled [0.1, 0.9]
    # We don't have coordinates, but cost encodes position
    # Lower cost means more central (closer to 0.5 in each dimension)
    
    # Create a cost-based prior: assume cost correlates with distance from center
    # cost = 1 + mean(x_scaled), so x_scaled_mean = cost - 1
    # For a point at center (0.5, 0.5), cost = 1.5
    # For corner (0, 0) or (1, 1), cost = 1.0 or 2.0
    # Actually: cost = 1 + (x1 + x2)/2 where x1, x2 in [0,1]
    # So cost ranges from 1.0 (both at 0) to 2.0 (both at 1)
    # Center (0.5, 0.5) has cost = 1.5
    
    # Prior probability that the minimum is near a point:
    # The minimum is uniformly distributed in inner 80%: [0.1, 0.9]^2
    # We can estimate proximity to center from cost
    # But this is indirect. Let's use a different approach.
    
    # Hybrid: combine EI and a cost-efficiency bonus
    # Points with lower cost per unit of expected improvement are better
    # Also, points with high uncertainty in low-cost regions are valuable
    
    # Cost efficiency: how much improvement per unit cost
    # But we need to account for the fact that low-cost points are more central
    # and thus more likely to be near the true minimum
    
    # Let's use a weighted combination:
    # - 70% weight on EI/cost (exploitation with cost awareness)
    # - 30% weight on sigma/cost (exploration in cheap regions)
    
    score_exploration = sigma / (cost + 0.5)
    
    # Normalize both scores to comparable scales
    ei_max = np.max(score_ei) if np.max(score_ei) > 0 else 1.0
    exp_max = np.max(score_exploration) if np.max(score_exploration) > 0 else 1.0
    
    score_ei_norm = score_ei / (ei_max + 1e-12)
    score_exp_norm = score_exploration / (exp_max + 1e-12)
    
    # Combined score
    score = 0.7 * score_ei_norm + 0.3 * score_exp_norm
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
