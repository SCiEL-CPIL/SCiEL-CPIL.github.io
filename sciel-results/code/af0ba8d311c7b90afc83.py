import numpy as np
import time

_START_TIME = time.time()
DEADLINE = 150.0
SAFETY_MARGIN = 2.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline check
    if time.time() > DEADLINE - SAFETY_MARGIN:
        # Fallback to simple Best Fit
        feasible = bins >= item
        scores = np.full_like(bins, -np.inf, dtype=float)
        if np.any(feasible):
            rem = bins[feasible] - item
            scores[feasible] = -rem
        return scores

    C = 100.0
    n_bins = len(bins)
    
    # Identify feasible bins
    feasible = bins >= item
    if not np.any(feasible):
        return np.full_like(bins, -np.inf, dtype=float)
    
    scores = np.full_like(bins, -np.inf, dtype=float)
    feasible_indices = np.where(feasible)[0]
    
    if len(feasible_indices) == 1:
        scores[feasible_indices[0]] = 0.0
        return scores
    
    # Composite scoring: combine multiple quality dimensions
    rem = bins[feasible] - item  # remaining capacity after placing item
    used = bins[feasible]        # current usage (before placing item)
    
    # Dimension 1: Best-fit component (prefer tighter fit)
    # Normalize remaining space: smaller rem is better
    best_fit_score = -rem / C
    
    # Dimension 2: Fragmentation penalty
    # Penalize bins that would have "awkward" remaining space
    # Very small remaining (<5%) or very large remaining (>40%) are suboptimal
    rem_frac = rem / C
    frag_penalty = np.where(
        (rem_frac < 0.05) | (rem_frac > 0.40),
        0.3 * np.abs(rem_frac - 0.25),  # distance from sweet spot ~25%
        0.0
    )
    
    # Dimension 3: Balance component
    # Prefer bins with moderate usage to avoid extreme empty or nearly-full bins
    used_frac = used / C
    balance_score = -np.abs(used_frac - 0.5) * 0.2
    
    # Dimension 4: Index preference (mild)
    # Slight preference for earlier bins to consolidate
    index_pref = -np.arange(n_bins)[feasible] * 0.001
    
    # Combine: weighted sum with best-fit dominant
    composite = (0.6 * best_fit_score - frag_penalty + balance_score + index_pref)
    
    scores[feasible] = composite
    
    return scores
