import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
DEADLINE = 148.0  # 150s limit, 2s safety margin
SAFETY_MARGIN = 2.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline check
    if time.time() > DEADLINE - SAFETY_MARGIN:
        # Fallback to simple Best Fit
        feasible = bins >= item
        scores = np.full_like(bins, -np.inf, dtype=float)
        if np.any(feasible):
            scores[feasible] = -(bins[feasible] - item)
        return scores

    C = 100.0
    n_bins = len(bins)
    
    # Identify feasible bins
    feasible = bins >= item
    if not np.any(feasible):
        return np.full_like(bins, -np.inf, dtype=float)
    
    scores = np.full_like(bins, -np.inf, dtype=float)
    b = bins[feasible]
    
    # Strategy: Smooth Adaptive Best Fit
    # The seed program uses pure Best Fit (minimize remaining space).
    # Previous attempts at hard switching to Worst Fit for small items failed (score < 1.0).
    # This suggests that for Weibull distributions, fragmentation from Worst Fit is costly.
    # However, pure Best Fit can lead to "dead" bins with very little remaining space
    # that can't fit most future items, while leaving other bins partially filled.
    
    # Hypothesis: A smooth transition that stays close to Best Fit but slightly
    # penalizes bins that are *almost* full (where the item would leave a tiny, useless
    # remainder) while still strongly favoring tight fits for larger items.
    # Alternatively, a "Ratio" based score: prioritize bins where the item fills a
    # significant fraction of the remaining capacity? No, Best Fit is already maximizing
    # the fraction of remaining capacity filled (1 - rem/C).
    
    # Let's look at the "Fluid Flow" / "Relative Waste" idea from the directive.
    # "prioritizing bins where the item constitutes a large fraction of the remaining capacity"
    # Score = item / (bins - item)? Or item / bins?
    # Best Fit minimizes (bins - item). This is equivalent to maximizing item / (bins - item)
    # for fixed item? No.
    # If we want to minimize relative waste, we might want to maximize the ratio of
    # filled space to total space? That's just load/C, which is constant for a given bin state.
    
    # Let's try a smooth surrogate for Best Fit that accounts for the "utility" of the remaining space.
    # If the remaining space is very small, it's useless. If it's large, it's useful.
    # Best Fit minimizes remaining space.
    # Let's try: Score = -remaining_space + lambda * (remaining_space / C)^2 ?
    # This would penalize large remainders more, pushing towards Best Fit even harder?
    # Or penalize small remainders?
    
    # Actually, the finding "Replacing the hard threshold with a smooth sigmoid transition that favors Best Fit for large items and becomes neutral... for small items" improved island 7 from 0.96 to 1.0.
    # This implies that for small items, being "neutral" (or slightly less Best-Fit-like) was better than pure Best Fit?
    # Or was it better than the previous *Worst Fit* strategy?
    # The text says: "avoids the fragmentation penalty of the previous Worst Fit strategy".
    # So the previous strategy was likely Best Fit for large, Worst Fit for small.
    # The improvement was moving from Worst Fit (for small) to Neutral (for small).
    # This suggests that for small items, Worst Fit is bad, and Best Fit might be okay, or Neutral is better.
    
    # Let's test a smooth blend between Best Fit and a "Neutral" (e.g., random or uniform) or slightly Worst-Fit-ish but not as extreme?
    # Actually, "Neutral" in a scoring context usually means score ~ 0 or constant.
    # If score is constant for small items, the choice among feasible bins is arbitrary (or based on tie-breaking).
    # Arbitrary choice for small items might be better than Best Fit?
    # Best Fit for small items packs them into already-full bins, which is generally good for minimizing bin count.
    # Why would Neutral be better?
    # Maybe because Best Fit for small items creates bins with *very* little space left, which are dead weight,
    # whereas putting them in "middle" bins keeps those bins more flexible?
    
    # Let's try a smooth transition:
    # t = item / C
    # For t near 1 (large items), use Best Fit: score = -rem
    # For t near 0 (small items), use a softer score, e.g., -rem * (1 - alpha * t) or similar?
    # Or blend between Best Fit (-rem) and Worst Fit (rem) with a weight based on item size?
    # The previous failure was a hard switch. A smooth sigmoid switch might work.
    
    # Let's define a weight w(t) that goes from 1 (Best Fit) to 0 (Neutral/Worst Fit) as t decreases?
    # If we blend Best Fit (-rem) and Worst Fit (rem):
    # score = w * (-rem) + (1-w) * (rem)
    # If w=1, score=-rem (Best Fit).
    # If w=0, score=rem (Worst Fit).
    # The finding says "becomes neutral (rather than Worst Fit)".
    # So maybe blend Best Fit (-rem) and 0?
    # score = w * (-rem)
    # If w=1, Best Fit. If w=0, all scores 0 (Neutral).
    # Let's try this.
    
    t = item / C
    # Sigmoid to transition from w=1 (large items) to w=0 (small items)?
    # Or the other way around?
    # Large items: Best Fit is crucial to avoid opening new bins unnecessarily or leaving large gaps.
    # Small items: Maybe Best Fit is too aggressive?
    
    # Let's use a sigmoid centered at some threshold, say 0.5.
    # But Weibull items are likely small on average?
    # Let's try a simple linear blend first, or a smooth step.
    # w = 1 - t  (linear: large items t~1 -> w~0? No, we want Best Fit for large).
    # w = t  (large items t~1 -> w~1 (Best Fit), small items t~0 -> w~0 (Neutral)).
    
    # Let's try w = t.
    # score = t * (-rem)
    # For large items (t~1), score ~ -rem (Best Fit).
    # For small items (t~0), score ~ 0 (Neutral).
    
    # Is "Neutral" (all 0) better than Best Fit for small items?
    # If all scores are 0, the evaluator might pick the first feasible bin? Or random?
    # The interface says "item goes to the highest-scoring bin". If tied, it's implementation dependent.
    # Usually, np.argmax returns the first occurrence.
    # So for small items, it would always pick the first feasible bin.
    # This is essentially "First Fit" for small items.
    # First Fit is often competitive.
    
    # Let's try this hypothesis:
    # score = t * (-(b - item))
    
    rem = b - item
    t = item / C
    scores[feasible] = t * (-rem)
    
    return scores
