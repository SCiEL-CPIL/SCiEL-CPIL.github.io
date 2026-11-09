import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    t0 = time.time()
    deadline = t0 + 115.0
    
    remaining = bins - item
    feasible = remaining >= 0
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(feasible):
        return scores
    
    rem = remaining[feasible]
    
    # Bimodal residual scoring:
    # Prefer placements where the item nearly fills the bin (ratio ~ 1.0)
    # OR leaves a large remainder (ratio small, preserving bin for future items).
    # Penalize moderate ratios that create unusable fragments.
    ratio = item / (rem + 1e-6)
    
    # Base score: mild preference for smaller remainders (best-fit component)
    # Use a gentler power to avoid over-penalizing larger remainders
    base = -np.sqrt(rem + 1e-6)
    
    # Strong fragmentation penalty: discourage ratio in [0.15, 0.60]
    # Wider and stronger than before to clearly create the bimodal preference
    center = 0.35
    width = 0.20
    penalty = -8.0 * np.exp(-((ratio - center) ** 2) / (2 * width ** 2))
    
    # Bonus for very high utilization (ratio close to 1.0) - fill the bin
    high_util_bonus = 3.0 * np.exp(-((ratio - 0.95) ** 2) / (2 * 0.05 ** 2))
    
    # Bonus for low utilization (ratio close to 0) - preserve bin space
    # This is the second mode of the bimodal distribution
    low_util_bonus = 2.0 * np.exp(-((ratio - 0.05) ** 2) / (2 * 0.10 ** 2))
    
    scores[feasible] = base + penalty + high_util_bonus + low_util_bonus
    
    if time.time() > deadline:
        return scores
        
    return scores
