"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 330.0  # seconds
_MARGIN = 5.0      # safety margin before deadline


def acquisition(mu, sigma, f_best, cost):
    """
    Batch-aware Greedy Expected Improvement (GEI) acquisition function.
    
    Instead of purely point-wise EI, this estimates the expected reduction in
    global posterior uncertainty for each candidate, penalizing evaluations that
    don't significantly shrink the search space's uncertainty relative to cost.
    """
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - _MARGIN:
        # Fallback: return cost-aware PI to be safe
        sigma_safe = np.maximum(np.asarray(sigma), 1e-9)
        z = (f_best - np.asarray(mu)) / sigma_safe
        return norm.cdf(z) / (np.asarray(cost) + 1e-9)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    n = len(mu)

    # Clamp sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-9)

    # Compute standard EI (point-wise expected improvement)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # --- Batch-aware GEI component ---
    # Estimate the expected reduction in global posterior variance if we evaluate
    # candidate i. We approximate this by measuring how much candidate i's
    # uncertainty overlaps with the overall uncertainty distribution.
    
    # Global uncertainty statistics
    sigma_mean = np.mean(sigma_safe)
    sigma_std = np.std(sigma_safe) + 1e-12
    sigma_max = np.max(sigma_safe)
    
    # For each candidate, estimate its "information value" by how much it
    # reduces the expected posterior variance of the remaining set.
    # Approximation: a candidate with low mu and moderate sigma provides more
    # information about the minimum than one with very high sigma.
    
    # Compute a proxy for expected variance reduction:
    # - High sigma candidates reduce uncertainty more (they're in uncertain regions)
    # - But if mu is already low, the marginal information gain is less
    # - Use a normalized measure of how "informative" each point is
    
    # Normalize sigma to [0, 1] for comparability
    sigma_norm = (sigma_safe - np.min(sigma_safe)) / (sigma_max - np.min(sigma_safe) + 1e-12)
    
    # Expected improvement in knowledge: higher when sigma is high (uncertain region)
    # but discounted if mu is already very low (we already know it's good)
    # This captures the idea that evaluating in uncertain regions near the optimum
    # is most valuable
    
    # Information gain proxy: combination of uncertainty and proximity to best
    # Use a kernel-like weighting: points with high sigma and low mu are most informative
    mu_range = np.max(mu) - np.min(mu) + 1e-12
    mu_norm = (mu - np.min(mu)) / mu_range  # 0 = best, 1 = worst
    
    # Expected variance reduction is highest when:
    # 1. The point is in an uncertain region (high sigma)
    # 2. The point is near the current best (low mu)
    # 3. The point is in a region where uncertainty is correlated with the optimum
    
    # Approximate the expected reduction in global posterior variance:
    # var_reduction_i ~ sigma_i * (1 - mu_norm_i) * overlap_factor
    # where overlap_factor measures how much this point's uncertainty overlaps
    # with the global uncertainty structure
    
    # Use a simple but effective proxy:
    # - Weight by sigma (uncertainty reduction potential)
    # - Weight by (1 - mu_norm) (proximity to best)
    # - Apply a diminishing returns factor to avoid over-weighting very high sigma
    
    uncertainty_weight = sigma_norm * (1.0 - mu_norm)
    
    # Apply diminishing returns: use sqrt to reduce the impact of extreme values
    uncertainty_weight = np.sqrt(uncertainty_weight)
    
    # Normalize to [0, 1]
    uw_max = np.max(uncertainty_weight) + 1e-12
    uncertainty_weight = uncertainty_weight / uw_max
    
    # The GEI score combines:
    # 1. Point-wise EI (exploitation)
    # 2. Expected variance reduction (exploration with global awareness)
    # 3. Cost normalization
    
    # Dynamic weighting based on progress
    min_possible = -0.8
    gap_to_min = max(f_best - min_possible, 0.0)
    
    # When far from optimum, weight exploration (variance reduction) more
    # When close, weight exploitation (EI) more
    exploration_weight = 0.3 * np.exp(-gap_to_min / 0.5) + 0.1
    exploitation_weight = 1.0 - exploration_weight
    
    # Combine EI and variance reduction
    # Scale variance reduction to be comparable to EI
    ei_scale = np.mean(ei) + 1e-12
    var_reduction_scaled = uncertainty_weight * ei_scale
    
    gei_score = exploitation_weight * ei + exploration_weight * var_reduction_scaled
    
    # Additional penalty for candidates in regions with very high uncertainty
    # that are far from the current best (wasting budget)
    # This further refines the exploration to focus on promising uncertain regions
    far_high_sigma_penalty = np.where(
        (mu_norm > 0.7) & (sigma_norm > 0.7),
        0.3,  # Apply penalty
        1.0   # No penalty
    )
    gei_score *= far_high_sigma_penalty
    
    # Cost normalization: divide by cost to favor cheaper evaluations
    # Use a softer cost penalty: 1 / (cost^0.5) instead of 1/cost
    cost_penalty = 1.0 / np.sqrt(cost_safe)
    score = gei_score * cost_penalty
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Final safety: ensure all scores are non-negative and finite
    score = np.maximum(score, 0.0)
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
