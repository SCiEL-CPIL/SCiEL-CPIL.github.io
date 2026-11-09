"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Bimodal residual scoring that prefers bins where the item fits
    with very little waste (near 0 residual) OR bins that are nearly full
    (preserving medium-space bins for future items). This balances tight
    packing with preserving flexibility.
    
    For a bin with remaining capacity r = bins[i]:
    - If r < item: not eligible (score = -inf)
    - residual = r - item
    - We want to prefer small residuals (tight fit) but also avoid leaving
      bins with "awkward" medium residuals that won't fit many future items.
    
    Score design:
    - Primary: minimize residual (best-fit)
    - Secondary: bonus for very tight fits (residual near 0)
    - Penalty: for residuals in a "dead zone" where the remaining space
      is unlikely to be useful for future items (Weibull items on [0,100])
    
    With Weibull distribution on [0,100], most items are small. The dead zone
    for residual is roughly (20, 40) where space is too small for many items
    but too large to be considered "tight". We penalize these.
    
    Actually, let's think more carefully. The key insight from the knowledge:
    "Bimodal residual scoring (preferring residuals near 0 or near capacity)
    may outperform pure best-fit by preserving bins with moderate space for
    future items"
    
    So we want to prefer:
    1. Residuals near 0 (tight fit)
    2. Residuals near capacity (i.e., the bin was nearly full before, so 
       after placing the item it's still nearly full - this means we're 
       filling up bins efficiently)
    
    Wait, "near capacity" for residual means residual is close to the bin 
    capacity, which means the bin was nearly empty. That doesn't make sense 
    for "preserving bins with moderate space".
    
    Let me reconsider. "Preserving bins with moderate space for future items"
    means we should NOT use bins with moderate space. We should use either:
    - Bins that become nearly full (small residual after placement)
    - Bins that were already nearly full (large residual, but the bin is 
      being finished off)
    
    Actually, I think the bimodal idea is:
    - Prefer bins where after placing the item, the residual is very small 
      (bin is nearly full) OR very large (bin still has lots of room, so 
      it's good for future items)
    - Avoid bins where the residual is in the middle (wasted space that 
      won't fit many future items)
    
    Let's define:
    - r = bins[i] - item (residual after placement)
    - If r < 0: not eligible
    - Score = -|r - target| where target adapts, or use a bimodal penalty
    
    A simpler approach: score = -(residual) + bonus_for_tight_fit
    where bonus_for_tight_fit is large when residual is very small.
    
    Or: score = -residual * (1 + alpha * exp(-beta * residual))
    This gives extra preference to small residuals.
    
    Let me try: score = -(residual) + gamma * exp(-delta * residual)
    The exponential term gives a big bonus for very small residuals.
    
    For Weibull on [0,100], typical items are small (mean ~20-30 for 
    shape=2, scale=100). So residuals of 0-10 are "tight", 10-50 are 
    "medium", 50+ are "large".
    
    Let's use: score = -residual + 10 * exp(-0.1 * residual)
    This gives:
    - residual=0: score = 10
    - residual=5: score = -5 + 10*0.607 = 1.07
    - residual=10: score = -10 + 10*0.368 = -6.32
    - residual=20: score = -20 + 10*0.135 = -18.65
    - residual=50: score = -50 + 10*0.0067 = -49.93
    
    This strongly prefers tight fits, which is similar to best-fit but with
    a stronger preference for very tight fits.
    
    Actually, the original best-fit is just -residual. The bimodal idea 
    suggests we should also prefer large residuals (bins that still have 
    lots of space). But that would be "first-fit" like behavior.
    
    Let me try a different approach: combine best-fit and first-fit with 
    a weighting that depends on the item size.
    
    For small items, best-fit is good (pack tightly).
    For large items, first-fit or best-fit-decreasing might be better.
    
    Since we don't know future items, let's use a heuristic:
    score = -residual + lambda * (bins[i] / 100) * some_factor
    
    Actually, let's just go with a tuned version of best-fit with a 
    tight-fit bonus. The measured result showed bimodal scoring improved 
    to 1.02, so let's try to capture that.
    
    Final design:
    - residual = bins - item
    - eligible = residual >= 0
    - score = -residual + tight_bonus
    - tight_bonus = A * exp(-B * residual)
    
    Let A=15, B=0.15. This gives strong preference for tight fits.
    """
    residual = bins - item
    eligible = residual >= 0
    
    # Start with -inf for ineligible bins
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    if not np.any(eligible):
        return scores
    
    res_elig = residual[eligible]
    
    # Base score: negative residual (best-fit)
    base = -res_elig
    
    # Tight-fit bonus: exponential decay
    tight_bonus = 15.0 * np.exp(-0.15 * res_elig)
    
    # Combined score
    scores[eligible] = base + tight_bonus
    
    return scores
# EVOLVE-BLOCK-END
