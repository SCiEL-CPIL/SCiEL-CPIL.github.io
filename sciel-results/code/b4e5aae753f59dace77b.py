"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    # Deadline guard: track elapsed time to ensure we finish well before the 300s limit
    start_time = time.time()
    deadline = 290.0  # Safety margin: finish by 290s

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid numerical issues
    sigma = np.maximum(sigma, 1e-12)
    
    # Check if we're approaching the deadline
    if time.time() - start_time > deadline:
        # Fallback: return a simple UCB score
        beta = 1.5
        score = -mu + beta * sigma
        score = np.where(np.isfinite(score), score, -np.inf)
        return score

    # Pareto-frontier filtering strategy:
    # 1. Mask out candidates whose predicted mean mu is worse than f_best by more than a tolerance
    # 2. Apply pure UCB only to the filtered subset
    
    # Tolerance: we keep candidates where mu <= f_best + tolerance
    # This restricts search to regions with plausible improvement
    tolerance = 0.05  # Small tolerance to allow some exploration near the boundary
    
    # Create mask: keep candidates where mu is close to or better than f_best
    mask = mu <= (f_best + tolerance)
    
    # If no candidates pass the filter (shouldn't happen often), relax the filter
    if not np.any(mask):
        # Use a more relaxed tolerance
        mask = mu <= (f_best + 0.5)
    
    # If still no candidates, use all (fallback)
    if not np.any(mask):
        mask = np.ones_like(mu, dtype=bool)
    
    # Compute UCB score for the filtered subset
    # UCB for minimization: score = -mu + beta * sigma
    # We want higher score to mean "evaluate next"
    
    # Adaptive beta: use a moderate exploration value
    # Since we're filtering to promising regions, we can be more aggressive with exploitation
    # but still maintain some exploration
    beta = 2.0
    
    # Initialize scores with -inf (will not be selected)
    score = np.full_like(mu, -np.inf, dtype=np.float64)
    
    # Compute UCB only for masked candidates
    score[mask] = -mu[mask] + beta * sigma[mask]
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Check deadline again
    if time.time() - start_time > deadline:
        # If we're close to deadline, return what we have
        return score
    
    return score
# EVOLVE-BLOCK-END
