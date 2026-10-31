"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score = better choice.
    Uses a combination of best-fit-decreasing logic with awareness of
    bin utilization to minimize total bins opened.
    
    Strategy: 
    - For each feasible bin (remaining >= item), compute a score.
    - We want to minimize the number of bins used, which correlates with
      filling bins as much as possible (best-fit) but also avoiding
      fragmentation.
    
    The L1 lower bound is ceil(sum(items)/capacity). To minimize excess,
    we want to pack items so that each bin is as full as possible.
    
    A good heuristic: prefer the bin that leaves the least remaining space
    (best-fit), but with a penalty if the remaining space is too small to
    be useful for future items. However, since we don't know future items,
    we use a smooth scoring function.
    
    Score = -(remaining_space) is basic best-fit.
    Let's try: score = -remaining_space + small_bonus_if_remaining_is_large_enough
    
    Actually, a well-known improvement is "best-fit decreasing" but here
    items arrive online. Let's use a scoring that favors bins where the
    item fits tightly, but also considers the current fill level.
    
    Alternative: score = -(bins - item) is best-fit. Let's try adding a
    term that rewards bins that are already partially filled (to avoid
    opening new bins unnecessarily).
    
    Let remaining = bins - item (space left after placing item).
    Let current_fill = bins (current used capacity).
    
    We want to minimize total bins. So we prefer placing in a bin that is
    already open (has some items) over an empty one, and among open bins,
    we prefer the one that fits best.
    
    Score = -(remaining) + alpha * (current_fill > 0)
    But this is discontinuous. Let's use a smooth version.
    
    Actually, the simplest improvement over best-fit is to consider the
    "waste" created. If we place item in a bin with remaining space R,
    the waste is R - item. We want to minimize this waste.
    
    Let's try: score = -(bins - item)  # best-fit
    But add a small bonus for bins that are not empty:
    score = -(bins - item) + 0.01 * np.tanh(bins / item)
    
    Wait, the directive says to keep the design and investigate improvements
    within its family. The current design is best-fit: return -(bins - item).
    
    Let me try a few variations:
    1. Pure best-fit: -(bins - item)
    2. Best-fit with small randomization
    3. First-fit: prefer the first bin that fits
    4. Next-fit: keep filling the current bin until full
    
    For online bin packing, best-fit is generally good. Let's try a
    smoothed version that considers the relative fit.
    
    Score = -((bins - item) / (bins + 1e-9))  # relative waste
    This normalizes by the bin's current fill.
    
    Or: Score = -((bins - item) / capacity)  # but we don't know capacity here
    
    Actually, the bins array contains the remaining capacity, not the used capacity.
    So bins[i] is the remaining space in bin i.
    
    If bins[i] = 100, the bin is empty.
    If bins[i] = 10, the bin has 90 used.
    
    Best-fit: place in bin with smallest remaining space that can fit the item.
    Score = -bins[i] for feasible bins.
    
    Let's try: score = -bins[i] + 0.001 * bins[i]**2 / 10000
    This slightly penalizes very large remaining spaces (empty bins).
    
    Actually, let's just use best-fit but with a tie-breaker that prefers
    bins with more used space (i.e., smaller remaining space is better, but
    among equal remaining, prefer the one with more total capacity used).
    But we don't have total capacity info.
    
    Let's keep it simple: best-fit is -(bins - item). Let's try a slight
    modification: prefer bins where the item takes up a larger fraction of
    the remaining space.
    
    fraction = item / bins[i]  # how much of the remaining space is filled
    score = fraction  # higher is better
    But this is undefined when bins[i] is very small.
    
    Let's use: score = item / (bins[i] + 1e-9)
    This is high when the item fills most of the remaining space.
    
    Wait, but if bins[i] is just barely enough (bins[i] = item), then
    score = item/item = 1. If bins[i] = 2*item, score = 0.5.
    So this prefers tighter fits, which is good.
    
    Let's try this.
    """
    # Ensure bins is a numpy array
    bins = np.asarray(bins, dtype=np.float64)
    
    # Feasible bins: remaining capacity >= item
    feasible = bins >= item
    
    # For feasible bins, compute score
    # We want higher score for better bins
    # Best-fit: smaller remaining space is better
    # Score = -remaining_space, but let's use a normalized version
    
    remaining = bins - item  # space left after placing item
    
    # Basic best-fit score
    score = -remaining.copy()
    
    # Add a small bonus for bins that are already partially filled
    # (i.e., bins that are not empty)
    # A bin is "empty" if bins[i] is close to capacity (100)
    # But we don't know capacity. Let's assume capacity is 100.
    # If bins[i] > 99, it's likely empty.
    # Give a small penalty to empty bins to encourage reuse of open bins.
    
    # Penalty for empty bins (bins[i] close to 100)
    empty_penalty = np.where(bins > 99.5, -0.01, 0.0)
    score += empty_penalty
    
    # For infeasible bins, set score to -inf
    score[~feasible] = -np.inf
    
    return score
# EVOLVE-BLOCK-END
