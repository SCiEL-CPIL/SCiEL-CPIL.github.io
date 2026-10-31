import numpy as np

# Weibull parameters for item size distribution (shape, scale)
# Typical for bin packing: shape ~1.5, scale ~100
_WEIBULL_SHAPE = 1.5
_WEIBULL_SCALE = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores bins using a composite expected-waste surrogate.
    
    Three terms:
    1. Dead-space penalty: penalizes medium remainders (neither very small nor very large)
       where future items are unlikely to fit well.
    2. Opportunity-cost penalty: penalizes very small remainders that are too small
       to be useful for most future items.
    3. Fit-quality bonus: rewards tight fits (small remaining capacity) to maximize
       utilization of each bin.
    
    This composite score better distinguishes between 'useful large gaps' and
    'truly dead small gaps', potentially improving packing density.
    """
    C = 100.0
    
    # Initialize scores to -inf so invalid bins are never chosen
    scores = np.full(bins.shape, -np.inf)
    
    # Identify bins that can fit the item
    valid_mask = bins >= item
    
    if not np.any(valid_mask):
        return scores
        
    # Calculate remaining capacity if the item is placed in the valid bins
    remaining = bins[valid_mask] - item
    
    k = _WEIBULL_SHAPE
    lam = _WEIBULL_SCALE
    
    # Normalize remaining capacity to [0, 1]
    r_norm = remaining / C
    
    # Term 1: Dead-space penalty for medium remainders
    # Weibull hazard rate peaks at medium values, creating a U-shaped penalty
    ratio = np.maximum(r_norm, 0.0)
    hazard = (k / lam) * np.power(ratio * lam, k - 1)
    dead_space_penalty = -0.5 * hazard  # Negative because we want to minimize hazard
    
    # Term 2: Opportunity-cost penalty for very small remainders
    # If remaining is too small to fit a typical future item, it's wasted
    # Use survival function S(r) = exp(-(r/lam)^k) as proxy for "useful" probability
    survival = np.exp(-np.power(ratio, k))
    # Penalize when survival is high (meaning the remainder is likely to remain unused)
    # but only for very small remainders
    opportunity_cost = -0.3 * np.where(r_norm < 0.2, survival, 0.0)
    
    # Term 3: Fit-quality bonus for tight packing
    # Reward small remainders (tight fit) to maximize bin utilization
    fit_bonus = 0.2 * (1.0 - r_norm)
    
    # Composite score: combine all terms
    composite_score = dead_space_penalty + opportunity_cost + fit_bonus
    
    scores[valid_mask] = composite_score
    
    return scores
