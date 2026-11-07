"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track time for deadline guard
_start_time = time.time()
_DEADLINE = 325.0  # 5 seconds before the 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Pareto-filtered UCB.
    
    Strategy:
    1. Mask out candidates whose predicted mean is worse than f_best by more
       than a tolerance (i.e., mu > f_best + tol). These are unlikely to improve.
    2. Apply a pure UCB score only to the remaining candidates.
    3. The UCB score is: -mu + beta * sigma, where beta is adaptive.
    
    This focuses search on regions with plausible improvement potential while
    still exploring high-uncertainty areas within that region.
    """
    # Deadline guard: if we're close to the time limit, return a simple score
    if time.time() > _DEADLINE:
        # Return -mu as a safe fallback (exploit best mean)
        sigma = np.maximum(sigma, 1e-9)
        return -mu
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Pareto filtering: mask out hopeless candidates
    # A candidate is "hopeless" if its predicted mean is much worse than f_best
    # Use a tolerance that scales with the current uncertainty level
    # When f_best is large (early), be more permissive; when small (late), be stricter
    
    # Tolerance: allow candidates whose mu is within a certain range of f_best
    # Use a dynamic tolerance based on the scale of the problem
    # Weierstrass function values are typically in [0, ~1.5]
    
    # Compute a scale-aware tolerance
    # If f_best is close to 0, use a tight tolerance
    # If f_best is large, use a wider tolerance
    
    # Use the 90th percentile of sigma as a scale reference
    sigma_scale = np.percentile(sigma, 90)
    
    # Tolerance: allow mu up to f_best + 2 * sigma_scale
    # This means we keep candidates that could potentially beat f_best
    # given the uncertainty in the GP prediction
    tol = 2.0 * sigma_scale
    mask = mu <= (f_best + tol)
    
    # If no candidates pass the filter (shouldn't happen, but be safe),
    # relax the filter to keep the best 50%
    if not np.any(mask):
        mu_sorted = np.sort(mu)
        threshold = mu_sorted[len(mu_sorted) // 2]
        mask = mu <= threshold
    
    # Adaptive beta for UCB
    # Beta should decrease as we get closer to the optimum
    # Use a logarithmic schedule based on f_best
    # When f_best is large, explore more (higher beta)
    # When f_best is small, exploit more (lower beta)
    
    # Scale beta inversely with how close we are to the minimum
    # f_best ranges from ~1.5 down to 0
    # Use: beta = base_beta * (1 + c * f_best)
    # where base_beta and c are tuned
    
    base_beta = 1.5
    c = 1.0
    beta = base_beta * (1.0 + c * abs(f_best))
    
    # Compute UCB score for minimization: -mu + beta * sigma
    ucb_score = -mu + beta * sigma
    
    # Apply the Pareto filter: set masked candidates to -inf
    score = np.where(mask, ucb_score, -np.inf)
    
    # Ensure all scores are finite (replace -inf with a very negative number)
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
