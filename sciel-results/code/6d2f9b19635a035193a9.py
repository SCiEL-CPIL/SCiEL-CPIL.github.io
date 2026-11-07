# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Quantile-based cost-effective acquisition with adaptive exploration/exploitation blend.
    
    Uses the 10th percentile of the predicted distribution (mu - 1.28*sigma) as a
    high-confidence lower bound on the objective, normalized by evaluation cost.
    Blends with an exploration term based on sigma to balance exploitation and
    exploration, with the blend weight adapting to the current best value.
    """
    # Deadline guard: track elapsed time for safety
    t0 = time.time()
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Quantile-based score: 10th percentile of predicted distribution
    # q_0.1 = mu - 1.2816 * sigma (1.2816 is the 10th percentile z-score)
    z_10 = -1.2815515655446004
    quantile = mu + z_10 * sigma_safe
    
    # Cost-adjusted quantile: we want to minimize quantile, so maximize -quantile/cost
    cost_adjusted_quantile = -quantile / cost_safe
    
    # Exploration term: sigma normalized by typical scale
    # Use a robust normalization based on f_best
    scale = max(abs(f_best), 1.0)
    exploration = sigma_safe / scale
    
    # Adaptive blend weight: 
    # When f_best is large (early in optimization), explore more
    # When f_best is small (late in optimization), exploit more
    # f_best for rosenbrock starts around 5-10 and goes toward 0
    # Use a sigmoid-like transition
    if f_best > 0.5:
        # Early phase: more exploration
        w_explore = 0.4
    elif f_best > 0.1:
        # Mid phase: balanced
        w_explore = 0.2
    else:
        # Late phase: more exploitation
        w_explore = 0.05
    
    # Combine: primary signal is cost-adjusted quantile, secondary is exploration
    # Normalize both to comparable scales
    caq_mean = np.mean(cost_adjusted_quantile)
    caq_std = np.std(cost_adjusted_quantile) + eps
    caq_norm = (cost_adjusted_quantile - caq_mean) / caq_std
    
    exp_mean = np.mean(exploration)
    exp_std = np.std(exploration) + eps
    exp_norm = (exploration - exp_mean) / exp_std
    
    scores = (1 - w_explore) * caq_norm + w_explore * exp_norm
    
    # Add a small constant to ensure positive scores for ranking stability
    # but the relative ordering is what matters
    scores = scores + 10.0
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
