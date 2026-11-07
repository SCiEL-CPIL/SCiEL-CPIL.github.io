"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = None


def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using information gain per unit cost (IG/C).
    
    Strategy:
    - Compute the predictive entropy reduction (information gain) for each candidate.
    - Normalize by cost to get IG/C, which prioritizes candidates that resolve
      uncertainty in low-cost regions.
    - Blend with Expected Improvement to maintain exploitation of known good regions.
    - Use a soft cost-aware weighting that heavily favors low-cost candidates.
    """
    # Deadline guard: if we're running long, return a safe fallback
    elapsed = _get_elapsed()
    if elapsed > 280:
        return np.zeros(len(mu))
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Clip EI to non-negative (it should be, but numerical issues can occur)
    ei = np.maximum(ei, 0.0)
    
    # Information Gain (IG): approximate via entropy reduction
    # The predictive entropy of a Gaussian is 0.5 * log(2*pi*e*sigma^2)
    # Reducing sigma by evaluating at x gives information gain proportional to
    # the log reduction in uncertainty. We use a proxy:
    # IG ≈ sigma * (some function of how much this point reduces local uncertainty)
    # A simple proxy: IG is proportional to sigma^2 (variance reduction)
    # But we want per-unit-cost, so we normalize by cost.
    
    # A better proxy for information gain: use the fact that evaluating at x
    # reduces the posterior variance. The information gain is related to
    # the mutual information between the observation and the function value.
    # For a GP, this is approximately proportional to sigma^2 in the prior,
    # but we only have the posterior. We use sigma as a proxy for remaining
    # uncertainty, and the information gain from evaluating is roughly
    # proportional to how much this point reduces the global uncertainty.
    
    # Simple and effective proxy: IG ≈ sigma (higher uncertainty = more to learn)
    # But we want to prioritize low-cost regions, so we divide by cost.
    ig_proxy = sigma_safe
    
    # Cost normalization: cost is in [1, 2] (since 1 + mean(x scaled to [0,1]))
    # Lower cost = cheaper evaluation
    cost_safe = np.maximum(cost, 1e-6)
    
    # IG per unit cost
    ig_per_cost = ig_proxy / cost_safe
    
    # Expected Improvement per unit cost
    ei_per_cost = ei / cost_safe
    
    # Blend IG/C and EI/C
    # When EI is high (we know a good region), exploit it
    # When EI is low but IG is high (uncertain region), explore it
    # The blend should favor IG/C in early stages and EI/C in later stages
    
    # Adaptive blend: use the ratio of mean EI to mean IG
    mean_ei = np.mean(ei)
    mean_ig = np.mean(ig_proxy)
    
    # When EI dominates, exploit more; when IG dominates, explore more
    # Normalize both to have similar scales
    ei_scale = np.max(ei) if np.max(ei) > 1e-12 else 1.0
    ig_scale = np.max(ig_proxy) if np.max(ig_proxy) > 1e-12 else 1.0
    
    ei_norm = ei / ei_scale
    ig_norm = ig_proxy / ig_scale
    
    # Weight: start with more exploration (IG), shift to exploitation (EI)
    # as the best value improves (f_best gets lower)
    # f_best starts high and decreases; use a sigmoid-like transition
    # Map f_best from [0, 100] to [0, 1] for the blend weight
    # Lower f_best = more exploitation
    f_best_clamped = np.clip(f_best, 0.0, 100.0)
    exploit_weight = 1.0 - f_best_clamped / 100.0  # 0 to 1
    exploit_weight = np.clip(exploit_weight, 0.1, 0.9)
    
    # Combine
    combined = exploit_weight * ei_norm + (1.0 - exploit_weight) * ig_norm
    
    # Cost-aware soft gate: strongly favor low-cost candidates
    # cost is in [1, 2], so cost-1 is in [0, 1]
    cost_offset = cost - 1.0  # in [0, 1]
    # Exponential decay: low cost gets high weight, high cost gets low weight
    # Use a sharp decay to heavily favor low-cost regions
    cost_weight = np.exp(-3.0 * cost_offset)  # ranges from 1.0 (cost=1) to ~0.05 (cost=2)
    
    # Apply cost weight multiplicatively
    scores = combined * cost_weight
    
    # Add a small bonus for very low-cost candidates to ensure exploration there
    low_cost_mask = cost < 1.2
    scores[low_cost_mask] += 0.01
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
