import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Deadline guard
    if time.time() - start_time > 1.0:
        # Fallback: Pure EI
        return ei
    
    # Structural change: Pareto-frontier guided exploration
    # Instead of fixed-weight blending, identify candidates that are
    # simultaneously good in exploitation (low mu) and exploration (high sigma),
    # then score based on a dynamic balance that adapts to convergence.
    
    # Normalize mu and sigma to [0, 1] for Pareto analysis
    mu_min, mu_max = np.min(mu), np.max(mu)
    mu_range = mu_max - mu_min
    if mu_range > 1e-12:
        mu_norm = (mu - mu_min) / mu_range  # 0 = best, 1 = worst
    else:
        mu_norm = np.zeros_like(mu)
        
    sigma_min, sigma_max = np.min(sigma_safe), np.max(sigma_safe)
    sigma_range = sigma_max - sigma_min
    if sigma_range > 1e-12:
        sigma_norm = (sigma_safe - sigma_min) / sigma_range  # 0 = least uncertain, 1 = most uncertain
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Find Pareto frontier: candidates where no other candidate is better in both
    # mu_norm (lower is better) and sigma_norm (higher is better for exploration)
    # A candidate is on the Pareto frontier if no other candidate has
    # mu_norm_i <= mu_norm_j AND sigma_norm_i >= sigma_norm_j (with at least one strict)
    
    n = len(mu)
    # For efficiency with 2000 points, use a simplified Pareto identification
    # Sort by mu_norm ascending, then find running max of sigma_norm
    order = np.argsort(mu_norm)
    mu_sorted = mu_norm[order]
    sigma_sorted = sigma_norm[order]
    
    # Running max of sigma_norm in sorted order
    running_max = np.maximum.accumulate(sigma_sorted)
    
    # A point is on the Pareto frontier if its sigma_norm equals the running max
    # at its position (and it's the first occurrence of that max)
    pareto_mask_sorted = np.zeros(n, dtype=bool)
    for i in range(n):
        if sigma_sorted[i] == running_max[i]:
            # Check if this is the first time we see this running max
            if i == 0 or sigma_sorted[i] > sigma_sorted[i-1]:
                pareto_mask_sorted[i] = True
    
    pareto_mask = np.zeros(n, dtype=bool)
    pareto_mask[order] = pareto_mask_sorted
    
    # Dynamic exploration-exploitation balance based on convergence
    # When f_best is close to 0 (optimal), favor exploitation more
    # When f_best is far from 0, favor exploration more
    convergence = np.exp(-f_best / 1.0)  # 1 when f_best=0, 0 when f_best is large
    alpha = 0.3 + 0.7 * convergence  # 0.3 to 1.0, more exploitation as we converge
    
    # Score: combination of exploitation (low mu) and exploration (high sigma)
    # But weighted by Pareto frontier membership
    exploitation_score = 1.0 - mu_norm  # Higher is better
    exploration_score = sigma_norm  # Higher is better
    
    base_scores = alpha * exploitation_score + (1 - alpha) * exploration_score
    
    # Boost scores for Pareto frontier candidates
    pareto_boost = 1.5
    scores = base_scores.copy()
    scores[pareto_mask] *= pareto_boost
    
    # Normalize scores to [0, 1] for consistency
    scores_min = np.min(scores)
    scores_max = np.max(scores)
    scores_range = scores_max - scores_min
    if scores_range > 1e-12:
        scores = (scores - scores_min) / scores_range
    else:
        scores = np.ones_like(scores)
    
    # Final deadline guard
    if time.time() - start_time > 2.0:
        # Fallback: Pure EI
        return ei
    
    return scores
