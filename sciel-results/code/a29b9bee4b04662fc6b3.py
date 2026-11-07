"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Heuristic using variance-aware weighted harmonic mean across constraints.
    
    For each item, compute prize/weight per constraint, then combine
    via a weighted harmonic mean where weights reflect constraint tightness
    (higher variance in weights indicates a more binding constraint).
    Also blends in a soft-min of per-constraint efficiencies to better
    approximate the actual bottleneck constraint.
    """
    # Per-constraint efficiency: prize / weight for each constraint
    # Shape: (5, 100)
    eff = prizes / (weights + 1e-9)
    
    # Constraint tightness: use variance as a proxy for how binding the
    # constraint is. Higher variance means items vary more in cost, so
    # the constraint is more likely to be a bottleneck (some items are
    # very expensive relative to others).
    # Shape: (5,)
    tightness_var = weights.var(axis=1)  # (5,)
    tightness_mean = weights.mean(axis=1)  # (5,)
    
    # Combine variance and mean: constraints with high mean AND high variance
    # are the most binding
    tightness = tightness_mean + tightness_var  # (5,)
    
    # Normalize to get weights
    w = tightness / (tightness.sum() + 1e-9)  # (5,)
    
    # Weighted harmonic mean: H = 1 / sum(w_i / eff_i)
    inv_eff = 1.0 / (eff + 1e-9)  # (5, 100)
    weighted_sum = (w[:, np.newaxis] * inv_eff).sum(axis=0)  # (100,)
    
    # Harmonic mean of efficiencies (higher = more attractive)
    h_mean = 1.0 / (weighted_sum + 1e-9)  # (100,)
    
    # Soft-min of efficiencies: approximates the minimum efficiency across
    # constraints using log-sum-exp with a temperature parameter.
    # Lower temperature = closer to true min.
    # soft_min(eff) = -logsumexp(-eff / T) * T
    T = 0.1  # temperature for soft-min
    neg_eff_scaled = -eff / T  # (5, 100)
    max_neg = neg_eff_scaled.max(axis=0, keepdims=True)  # (1, 100)
    log_sum_exp = np.log(np.sum(np.exp(neg_eff_scaled - max_neg), axis=0)) + max_neg.squeeze(0)
    soft_min_eff = -log_sum_exp * T  # (100,)
    
    # Blend harmonic mean and soft-min: the harmonic mean captures overall
    # efficiency, while soft-min captures the worst-case constraint.
    # Use a weighted blend with more weight on harmonic mean (it's smoother)
    alpha = 0.7  # weight for harmonic mean
    blended = alpha * h_mean + (1 - alpha) * soft_min_eff
    
    # Scale by mean prize to keep magnitude reasonable
    scale = prizes.mean()
    result = blended * scale
    
    # Ensure positivity and finite
    result = np.where(np.isfinite(result) & (result > 0), result, 1e-9)
    
    return result
# EVOLVE-BLOCK-END
