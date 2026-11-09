"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # bins is the array of remaining capacities for each bin.
    # The item goes to the highest-scoring bin among those with enough capacity.
    # We use a First-Fit Decreasing inspired heuristic adapted for online setting:
    # Prefer bins where the item fits and leaves a residual that is "close" to a
    # multiple of a typical item size, but since we don't know future items,
    # we use a combination of best-fit and a penalty for very small residuals.
    
    remaining = bins
    capacity = 100.0  # Fixed capacity per task description
    
    # Only consider bins that can fit the item
    mask = remaining >= item
    if not np.any(mask):
        # No bin can fit the item; return all -inf (should not happen if problem is feasible)
        return np.full_like(remaining, -np.inf)
    
    # For bins that can fit the item, compute the residual after placing the item
    residual = remaining[mask] - item
    
    # Score: We want to prefer bins that result in a residual that is either:
    # 1. Very small (best-fit) to minimize waste
    # 2. Or close to capacity (to leave room for large items)
    # A good heuristic: minimize the residual, but penalize residuals that are
    # "awkward" (e.g., too small to fit most future items but not zero).
    
    # Simple strong heuristic: Best-Fit (minimize residual) is already good.
    # Let's try a variation: prefer residuals that are as small as possible,
    # but with a slight bias toward bins that are already "fuller" to consolidate.
    
    # Alternative: Use a score that rewards bins where the new fill percentage
    # is high, but also considers the absolute residual.
    
    # Let's try: score = -residual (best-fit) but with a tie-breaking that prefers
    # bins with larger original capacity usage.
    
    # Actually, let's try a different family: "Worst-Fit" variant.
    # Worst-fit places the item in the bin with the most remaining space after placement.
    # This can be better for online packing as it preserves bins with small space
    # for small future items.
    
    # Score for worst-fit: maximize residual
    score = residual.copy()
    
    # But pure worst-fit can be bad. Let's blend best-fit and worst-fit.
    # Actually, research shows that Best-Fit is generally strong for online bin packing.
    # Let's try a "Best-Fit Decreasing" like approach even though it's online.
    
    # Another idea: prefer bins where the residual is closest to 0 OR closest to capacity.
    # This creates a bimodal preference.
    
    # Let's try: score = -min(residual, capacity - residual)
    # This prefers residuals that are either very small or very large.
    
    dist_to_zero = residual
    dist_to_cap = capacity - residual
    score = -np.minimum(dist_to_zero, dist_to_cap)
    
    # Map back to full array
    full_score = np.full_like(remaining, -np.inf)
    full_score[mask] = score
    
    return full_score
# EVOLVE-BLOCK-END
