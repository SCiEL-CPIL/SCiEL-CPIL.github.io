"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    """
    Compute an item-by-item attractiveness matrix for placing item j right after
    item i in the same bin. Larger values mean more attractive.

    The heuristic strongly favors tight fits (small remaining space after placing
    both items) and penalizes loose fits. Items that don't fit together get 1e-9.
    """
    n = len(demands)
    # Pairwise sum of demands
    pair_sum = demands[:, None] + demands[None, :]
    
    # Remaining capacity after placing both items
    remaining = capacity - pair_sum
    
    # Only valid if both items fit together
    fits = pair_sum <= capacity
    
    # For fitting pairs, we want to reward tight fits (small remaining).
    # Use inverse squared of remaining space (with a small epsilon to avoid div by zero
    # when remaining is exactly 0, which is the best case).
    # If remaining is 0, that's a perfect fit -> very high attractiveness.
    # We use 1 / (1 + remaining^2) to keep values bounded and positive.
    # Actually, let's use a sharper decay: 1 / (1 + (remaining / capacity)^2 * scale)
    # Let's think about the range: remaining can be 0 to capacity-40 (min demand is 20, so min pair sum is 40).
    # capacity = 150, so remaining ranges from 0 to 110.
    # We want remaining=0 to be very attractive, remaining=110 to be least attractive.
    
    # Strategy: attractiveness = 1 / (1 + (remaining / capacity)^2 * k)
    # With k=1, remaining=0 -> 1.0, remaining=150 -> 1/2=0.5. That's not sharp enough.
    # Let's use k=10: remaining=0 -> 1.0, remaining=75 (half cap) -> 1/(1+10*0.25)=1/3.5~0.29, remaining=150 -> 1/11~0.09.
    # Even sharper: k=100: remaining=0 -> 1.0, remaining=37.5 -> 1/(1+100*0.0625)=1/7.25~0.14.
    
    # Alternative: use exponential decay based on remaining space.
    # attractiveness = exp(-remaining / tau) where tau is a temperature.
    # smaller tau = sharper preference for tight fits.
    
    # Let's combine: for fitting pairs, use 1/(1 + remaining^2) scaled appropriately.
    # remaining ranges 0 to ~110. remaining^2 ranges 0 to 12100.
    # 1/(1+0) = 1, 1/(1+12100) ~ 0.00008. That's very sharp.
    # But we need to make sure non-fitting pairs get 1e-9, which is smaller than 0.00008.
    # So 1e-9 is fine as the floor for non-fitting.
    
    # Let's use: for fitting pairs, attractiveness = 1.0 / (1.0 + remaining**2)
    # This gives a strong preference for tight fits.
    
    # However, we should also consider the absolute size of the items.
    # Placing two large items that fit tightly is better than two small items with lots of space.
    # The remaining space already captures this: large items -> small remaining -> high attractiveness.
    
    # Let's also add a small bonus for the sum of items being large (to fill bins).
    # attractiveness = (pair_sum / capacity) / (1.0 + remaining**2)
    # This normalizes by the fraction of the bin filled.
    
    # Actually, let's keep it simple and effective:
    # For fitting pairs: 1.0 / (1.0 + (remaining / 10.0)**2)
    # This makes the decay scale relative to a typical remaining space of ~10.
    
    # Let me reconsider. The key insight from the knowledge is:
    # "Sharper decay for waste (squared inverse) strongly prioritizes tight fits"
    # So 1/(1 + remaining^2) or similar is good.
    
    # Let's use: for fitting pairs, attractiveness = 1.0 / (1.0 + remaining**2 / 100.0)
    # This normalizes the squared remaining by 100 to keep values in a reasonable range.
    # remaining=0 -> 1.0
    # remaining=10 -> 1/(1+1) = 0.5
    # remaining=50 -> 1/(1+25) = 0.038
    # remaining=100 -> 1/(1+100) = 0.0099
    
    # This seems reasonable. Let's go with this.
    
    result = np.full((n, n), 1e-9, dtype=np.float64)
    result[fits] = 1.0 / (1.0 + (remaining[fits] ** 2) / 100.0)
    
    return result
# EVOLVE-BLOCK-END
