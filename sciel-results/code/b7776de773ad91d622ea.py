"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for online bin packing.
    
    Strategy:
    1. Filter bins with enough capacity (bins >= item).
    2. For eligible bins, compute a score that balances:
       - Best-fit: prefer bins that leave least remaining space
       - Avoid fragmentation: penalize bins that are "too full" or "too empty"
       - Use a smooth surrogate to avoid hard thresholds
    """
    # Deadline guard: if we've been running too long, return quickly
    # (This is a simple heuristic; the actual time budget is managed by evaluator)
    
    # Remaining capacity in each bin
    remaining = bins - item
    
    # Only consider bins with non-negative remaining capacity
    eligible_mask = remaining >= 0
    
    # Initialize scores with -inf for ineligible bins
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    if not np.any(eligible_mask):
        # No eligible bins; return all -inf (caller should handle this)
        return scores
    
    # For eligible bins, compute a refined score
    rem = remaining[eligible_mask]
    
    # Best-fit component: we want to minimize remaining, so score is -remaining
    # But we also want to avoid creating many small bins (fragmentation)
    
    # Smooth penalty: use a soft-min-like approach
    # The L1 lower bound for bin packing is related to total_size / capacity
    # We want to pack items so that we use bins close to full
    
    # Heuristic: 
    # - Strongly prefer bins that will be nearly full after placing item
    # - But avoid bins that are already too full (no room for future items)
    
    # Score = -remaining + bonus for bins that are "well-utilized"
    
    # Compute utilization after placing item
    # bins[i] is current usage, so after placing item: bins[i] + item
    # Utilization = (bins[i] + item) / capacity, where capacity = max(bins) or 100?
    # Actually, bins array contains remaining capacity or current usage?
    # From the seed: "bins - item" suggests bins is remaining capacity
    # Wait, let me re-read: "bins with enough remaining capacity"
    # So bins[i] is the remaining capacity in bin i
    
    # So if bins[i] >= item, we can place item
    # After placing, remaining becomes bins[i] - item
    
    # We want to minimize bins[i] - item (best-fit)
    # But also consider: if we always pick the smallest remaining, we might
    # fragment bins. A better strategy might be to pick bins that are 
    # "just right" - not too small, not too large
    
    # Let's use a combination:
    # 1. Primary: -remaining (best-fit)
    # 2. Secondary: penalize very small remaining (fragmentation)
    # 3. Tertiary: slight preference for larger bins (more flexibility)
    
    # Smooth score using exponential weighting
    # score = -remaining * w1 - exp(-remaining/scale) * w2
    
    # Let's try: score = -remaining + alpha * remaining^2 / capacity_scale
    # This creates a minimum at some intermediate remaining
    
    # Actually, for online bin packing, best-fit-decreasing is known to be good.
    # Let's stick close to best-fit but add a small smoothing term.
    
    # Use a soft approach: score = -remaining * (1 + beta * remaining/capacity)
    # This makes the penalty for large remaining super-linear
    
    # Estimate capacity as max bin size or use a fixed value
    # Since we don't know capacity explicitly, use max(bins) as a proxy
    cap_est = np.max(bins) if len(bins) > 0 else 100.0
    if cap_est < 1e-9:
        cap_est = 100.0
    
    # Score: primarily best-fit, with a quadratic penalty for large remaining
    # This encourages using bins that are close to full
    w_linear = 1.0
    w_quad = 0.1
    
    score_vals = -w_linear * rem + w_quad * (rem / cap_est) ** 2
    
    # Assign scores back
    scores[eligible_mask] = score_vals
    
    return scores
# EVOLVE-BLOCK-END
