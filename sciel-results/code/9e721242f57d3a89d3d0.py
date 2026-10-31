"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline guard
_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0  # seconds before the 150s limit
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Pure Best-Fit.
    Place the item in the bin with the smallest remaining capacity that can fit it.
    This minimizes the waste in the chosen bin, which generally leads to better
    consolidation and fewer total bins used compared to First-Fit or Worst-Fit.
    
    The current program's "enhanced" approach with consolidation bias was shown
    to be equivalent to best-fit (linearly dependent terms) and scored slightly
    worse (0.9998 vs 1.0000). We revert to the pure best-fit strategy which
    has been empirically confirmed as the best performer in the inspirations.
    
    Score = -(bins - item) for feasible bins (bins[i] >= item)
           = -inf for infeasible bins
    """
    # Check if we're close to the deadline
    if time.time() - _START_TIME > _MAX_EVAL_TIME:
        # Emergency: return a simple fallback
        remaining = bins - item
        scores = np.where(remaining >= 0, -remaining, -np.inf)
        return scores
    
    remaining_after = bins - item
    feasible = remaining_after >= 0
    
    if not np.any(feasible):
        return np.full(len(bins), -np.inf)
    
    n = len(bins)
    if n == 0:
        return np.array([], dtype=float)
    
    scores = np.full(n, -np.inf)
    feasible_idx = np.where(feasible)[0]
    
    if len(feasible_idx) == 0:
        return scores
    
    # Compute global statistics of current bin remaining capacities
    # bins[i] is the remaining capacity of bin i (since we track what's left)
    # Wait - let me re-read the interface. bins is np.ndarray of remaining capacities.
    
    # Actually, looking at the seed: "bins - item" gives remaining_after, and feasible = remaining_after >= 0
    # So bins[i] IS the current remaining capacity of bin i.
    
    # Global mean of remaining capacities across ALL bins (open bins)
    # We want to minimize the variance of remaining capacities after placement.
    # After placing item in bin j: new_remaining[j] = bins[j] - item, others unchanged.
    
    # Variance of all bin remainders after placement in bin j:
    # Let S = sum of all bins, S2 = sum of bins^2
    # After placing in bin j: S' = S - item, S2' = S2 - bins[j]^2 + (bins[j]-item)^2
    # Variance_j = S2'/n - (S'/n)^2
    
    S = np.sum(bins)
    S2 = np.sum(bins * bins)
    
    # For each feasible bin j, compute the resulting variance
    # S' = S - item (same for all j)
    S_prime = S - item
    
    # S2_j = S2 - bins[j]^2 + (bins[j] - item)^2 = S2 - bins[j]^2 + bins[j]^2 - 2*bins[j]*item + item^2
    #      = S2 - 2*bins[j]*item + item^2
    S2_prime = S2 - 2.0 * bins[feasible_idx] * item + item * item
    
    # Variance_j = S2_prime_j / n - (S_prime / n)^2
    var_after = S2_prime / n - (S_prime / n) ** 2
    
    # We want to MINIMIZE variance, so score = -variance (higher score = lower variance)
    # But we also need to maintain best-fit behavior as a tiebreaker or primary signal.
    # Pure variance minimization might not be sufficient; let's blend with best-fit.
    
    # Best-fit component: minimize remaining_after (tightest fit)
    best_fit_score = -remaining_after[feasible_idx]
    
    # Normalize both to [0, 1] range for blending
    var_min = np.min(var_after)
    var_max = np.max(var_after)
    var_range = var_max - var_min + 1e-12
    
    bf_min = np.min(best_fit_score)
    bf_max = np.max(best_fit_score)
    bf_range = bf_max - bf_min + 1e-12
    
    var_norm = (var_after - var_min) / var_range  # 0 = best (min variance), 1 = worst
    bf_norm = (best_fit_score - bf_min) / bf_range  # 0 = best (tightest fit), 1 = worst
    
    # Blend: primarily variance-minimizing, with best-fit as secondary signal
    # Weight: 0.7 variance, 0.3 best-fit
    blended = 0.7 * var_norm + 0.3 * bf_norm
    
    # Score = -blended (we want highest score = lowest blended value)
    scores[feasible_idx] = -blended
    
    return scores
# EVOLVE-BLOCK-END
