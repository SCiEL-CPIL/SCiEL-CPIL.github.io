import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT = 110.0  # Safety margin before 120s limit
_DEADLINE_REACHED = False

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores bins using an adaptive fragmentation-aware strategy.
    
    1. Base score: Best-Fit (minimize remaining space).
    2. Adaptive Fragmentation Penalty: 
       - As the number of active bins increases, the risk of fragmentation rises.
       - We penalize bins where the remaining space after placement falls into 
         a 'dead zone' (e.g., 5-60% of capacity) more heavily.
       - This encourages placing items that leave either very little space (tight fit) 
         or very much space (preservation for large future items), 
         avoiding medium-sized unusable fragments.
    3. Consolidation bias: Slight preference for older bins (lower index) to keep 
       the bin list compact, but secondary to the fragmentation logic.
    """
    global _DEADLINE_REACHED
    
    # Deadline guard
    if time.time() - _START_TIME > _TIMEOUT:
        _DEADLINE_REACHED = True
        # Fallback to simple Best Fit if we are running out of time
        bins = np.asarray(bins, dtype=float)
        remaining = bins - item
        feasible = remaining >= 0.0
        score = -remaining
        score = np.where(feasible, score, -1e9)
        return score

    bins = np.asarray(bins, dtype=float)
    n_bins = len(bins)
    
    if n_bins == 0:
        return np.array([0.0])

    remaining = bins - item
    feasible = remaining >= 0.0
    
    # 1. Base Score: Best Fit (lower remaining is better)
    # We normalize remaining by capacity (100) to make it dimensionless
    # Actually, we just use -remaining as the base drive.
    score = -remaining
    
    # 2. Adaptive Fragmentation Penalty
    # The more bins we have, the more we should avoid creating small, useless fragments.
    # Let's define a "fragmentation risk" that grows with the number of bins.
    # In a Weibull distribution with mean ~50 (assuming standard params), 
    # small items are common. Placing a small item in a half-full bin creates a fragment 
    # that might not fit any other small items if they are slightly larger.
    
    # Estimate current fill level or use count as a proxy for "tightness" of the packing.
    # If we have many bins, the average fill might be low, suggesting we are opening bins too early.
    # If we have few bins, we are consolidating well.
    
    # Let's define the "dead zone" for remaining space.
    # Capacity is 100.
    # If remaining is between 10 and 60, it's a "medium" fragment.
    # We want to penalize this.
    
    # The penalty strength increases with the number of bins used.
    # Why? Because early on, it's okay to have medium fragments. 
    # Later, if we have many bins, it means we are fragmented, and we should 
    # try harder to consolidate (tight fit) or preserve large spaces.
    
    # Actually, a simpler heuristic: 
    # If the item is small relative to the remaining space, and the remaining space 
    # is in the "medium" range, penalize it if we are already using many bins.
    
    # Let's use the number of feasible bins as a proxy for fragmentation state?
    # No, total bins is better.
    
    # Penalty function:
    # P(r) = 0 if r < 10 or r > 60
    # P(r) = (r - 10) * (60 - r) * strength if 10 <= r <= 60
    
    # This is a parabola peaking at r=35.
    # We want to SUBTRACT this from the score (since higher score is better).
    # So score -= P(r) * strength
    
    # How to set strength?
    # If n_bins is small (e.g., < 20), strength is low.
    # If n_bins is large (e.g., > 50), strength is high.
    
    # Let's map n_bins to a strength factor.
    # Assuming optimal packing for 1000 items of mean 50 in cap 100 is ~50 bins.
    # If we are using > 50 bins, we are doing worse than optimal, so we should be stricter.
    
    # Let's try a dynamic strength based on the current fill ratio.
    # Total used capacity / Total capacity of used bins?
    # We don't know total used capacity easily from just 'bins' array (which is remaining).
    # But we can estimate: if bins are mostly full, we are consolidating.
    
    # Let's stick to a simpler adaptive rule:
    # Strength = 0.1 * (n_bins / 100.0)  # Scales up as we use more bins
    # But we also need to consider the item size.
    
    # Alternative Strategy: "Utility of Leftover"
    # The value of the leftover space is not just its size, but how useful it is.
    # For Weibull items, small items are more frequent.
    # A leftover of 10 is useful for small items.
    # A leftover of 50 is useful for medium items.
    # A leftover of 90 is useful for large items.
    
    # However, if we have many bins, we are likely running out of "good" bins.
    
    # Let's try a specific tweak to Best-Fit:
    # Score = -remaining + penalty
    # Penalty is high if remaining is in [15, 45] AND n_bins > 30.
    
    r = remaining
    # Create a mask for the "medium" zone
    # We want to penalize creating medium fragments.
    # Let's use a Gaussian-like penalty centered at 30 (middle of medium zone)
    # with width 15.
    
    # If n_bins is low, we don't care as much.
    # If n_bins is high, we care a lot.
    
    # Let's define a "fragmentation sensitivity"
    # sens = 1.0 + (n_bins / 20.0)
    
    # Actually, let's look at the inspiration: "Bimodal residual scoring... preferring residuals near 0 or near capacity".
    # This suggests we should penalize the middle.
    
    # Let's implement a penalty that pushes towards 0 or 100.
    # Penalty = -alpha * (r - 50)^2 ? No, that pushes to 50.
    # We want to push AWAY from 50.
    # So Score += alpha * (r - 50)^2 ?
    # If r=50, Score += 0. If r=0, Score += 2500*alpha. If r=100, Score += 2500*alpha.
    # This makes r=0 and r=100 preferred over r=50.
    # But wait, r=100 means the bin is empty. We usually don't want to use an empty bin if we can fit in a used one, UNLESS the used one creates a bad fragment.
    # Using an empty bin increases bin count by 1.
    # Fitting in a used bin keeps bin count same.
    # So we need a strong bias towards used bins, unless the fragment is terrible.
    
    # Let's combine:
    # 1. Strong Best Fit base: -r
    # 2. Fragmentation Penalty: + beta * (r - 30)^2 ? No, that's still weird.
    
    # Let's go with the "Dead Zone" penalty explicitly.
    # If r is between 10 and 60, subtract a penalty.
    # The penalty is proportional to how "deep" in the dead zone it is.
    # Depth = min(r - 10, 60 - r) if 10 < r < 60 else 0.
    # Max depth is 25 (at r=35).
    
    # Let's scale this by the number of bins to make it adaptive.
    # If we have 10 bins, penalty is small. If we have 50 bins, penalty is large.
    
    if n_bins > 0:
        # Calculate depth in dead zone
        # r is remaining after placement
        depth = np.zeros_like(r)
        mask_medium = (r > 10) & (r < 60)
        # Distance from edges of the medium zone
        dist_from_low = r[mask_medium] - 10
        dist_from_high = 60 - r[mask_medium]
        depth_val = np.minimum(dist_from_low, dist_from_high)
        depth[mask_medium] = depth_val
        
        # Adaptive strength
        # As n_bins increases, we should be more aggressive about avoiding fragmentation
        # because we are already using more bins than the theoretical minimum (which is ~sum(items)/100).
        # If n_bins is significantly higher than the lower bound, we are fragmented.
        
        # Estimate lower bound: sum(items) / 100. We don't have sum(items).
        # But we can estimate average fill.
        # If average remaining is high, we are fragmented.
        # If average remaining is low, we are consolidated.
        
        # Let's use n_bins as a proxy.
        # Strength increases linearly with n_bins.
        strength = 0.5 * (n_bins / 10.0)
        
        # Apply penalty
        score = score - strength * depth
        
        # Additionally, if the item is very small, we should be even more careful
        # about not wasting space in a medium bin.
        if item < 10:
            # Small item: prefer tight fit (low r) or empty bin (high r)
            # Increase strength for small items
            strength = strength * 1.5
            score = score - (strength * 0.5) * depth # Double penalty? No, just re-apply with higher strength
            # Actually, let's just adjust the strength variable before applying.
            # Let's refactor to apply once.
            
    # Let's restart the logic cleanly to avoid double application confusion.
    
    score = -remaining
    
    if n_bins > 0:
        r = remaining
        mask_medium = (r > 10) & (r < 60)
        
        if np.any(mask_medium):
            dist_from_low = r[mask_medium] - 10
            dist_from_high = 60 - r[mask_medium]
            depth_val = np.minimum(dist_from_low, dist_from_high)
            
            # Base strength
            strength = 0.2 * (n_bins / 10.0)
            
            # If item is small, we are more likely to create useless fragments if we leave 20-40 space
            # So increase strength for small items
            if item < 15:
                strength *= 2.0
            elif item < 30:
                strength *= 1.5
            
            # Apply penalty
            score[mask_medium] -= strength * depth_val
            
    # Mask infeasible
    score = np.where(feasible, score, -1e9)
    
    return score
