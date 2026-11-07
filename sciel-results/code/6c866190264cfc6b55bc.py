"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid approach:
    - Expected Improvement (EI) for exploitation
    - Inverse-cost weighted uncertainty for exploration
    This creates a non-uniform exploration landscape that prioritizes
    high-uncertainty, low-cost regions early on.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a safe fallback: just exploit the best mean
        scores = -mu / cost
        if not np.all(np.isfinite(scores)):
            scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Clip f_best for numerical stability
    f_best_clipped = np.clip(f_best, 0.0, 100.0)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Handle sigma near zero to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best_clipped - mu) / sigma_safe
    
    # Standard normal CDF and PDF using scipy
    from scipy.stats import norm
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    EI = (f_best_clipped - mu) * Phi + sigma_safe * phi
    
    # Cost-aware weighting: prioritize low-cost evaluations
    # Use inverse cost to weight the acquisition score
    # This encourages the algorithm to explore cheap regions more aggressively
    
    # Adaptive exploration weight: decrease as we get closer to the optimum
    # When f_best is large (early), explore more; when f_best ~ 0, exploit more
    exploration_weight = 0.5 + 0.5 * np.exp(-f_best_clipped / 5.0)
    
    # Uncertainty term weighted by inverse cost
    # This creates a non-uniform exploration landscape
    uncertainty_term = exploration_weight * sigma_safe / cost
    
    # Combine EI and uncertainty term
    # The EI term drives exploitation, uncertainty term drives exploration
    scores = EI + uncertainty_term
    
    # Normalize scores to have zero mean and unit variance for stability
    score_mean = np.mean(scores)
    score_std = np.std(scores)
    if score_std > 1e-10:
        scores = (scores - score_mean) / score_std
    else:
        scores = scores - score_mean
    
    # Safety check for finite values
    if not np.all(np.isfinite(scores)):
        scores = np.where(np.isfinite(scores), scores, 0.0)
        
    return scores
# EVOLVE-BLOCK-END
