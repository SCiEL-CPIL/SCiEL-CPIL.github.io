# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization (minimization).
    Uses a structural change: instead of EI/cost + exploration, we use
    Lower Confidence Bound (LCB) with cost-awareness.
    
    score(x) = -LCB(x) / cost(x)
    where LCB(x) = mu(x) - kappa * sigma(x)
    
    This is a different modeling family (optimistic uncertainty bound vs. expected improvement).
    """
    t_start = time.time()

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    n = mu.size

    # Numerical guards
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)

    # Adaptive kappa based on uncertainty and cost
    # Use a fixed kappa for LCB, but normalize by cost
    # Typical approach: kappa ~ 1-2 for standard LCB
    # We'll use kappa = 1.0 as a reasonable default for exploration
    kappa = 1.0

    # LCB: mu - kappa * sigma (lower confidence bound for minimization)
    lcb = mu - kappa * sigma_safe

    # Cost-aware score: we want low LCB (good) and low cost (good)
    # So we minimize LCB * cost, or equivalently maximize -LCB / cost
    # But we need to be careful: if LCB is negative, -LCB is positive
    # Let's use: score = -(mu - kappa*sigma) / cost = (kappa*sigma - mu) / cost
    # This rewards low mu and high sigma (exploration), penalized by cost
    
    scores = (kappa * sigma_safe - mu) / cost_safe

    # Alternative: use a combination that ensures exploration
    # Let's also consider the probability of improvement as a weighting
    # P(y < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -40.0, 40.0)
    p_improve = norm.cdf(z)
    
    # Weight the LCB score by probability of improvement
    # This makes the acquisition more focused on regions likely to improve
    scores_weighted = scores * p_improve

    # Combine: use a blend of pure LCB/cost and probability-weighted version
    # Let's use the probability-weighted version as it's more targeted
    final_scores = scores_weighted

    # Ensure all values are finite (higher = evaluate next)
    final_scores = np.where(np.isfinite(final_scores), final_scores, 0.0)

    # Deadline guard (safety margin before the 330 s kill)
    if (time.time() - t_start) > 300.0:
        return final_scores

    return final_scores
# EVOLVE-BLOCK-END
