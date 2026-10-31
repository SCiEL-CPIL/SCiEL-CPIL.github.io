"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 140.0  # safety margin before 150s kill
_START_TIME = time.time()


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline guard: if we're too close to the limit, return a safe default
    if time.time() - _START_TIME > _DEADLINE:
        valid = bins >= item
        if not np.any(valid):
            return np.full_like(bins, -np.inf)
        scores = -(bins - item)
        scores[~valid] = -np.inf
        return scores

    # Determine which bins have enough remaining capacity
    valid = bins >= item
    
    if not np.any(valid):
        return np.full_like(bins, -np.inf)
        
    # New Strategy: Dynamic Congestion-Based Scoring
    # Instead of static rules (Best Fit vs Worst Fit), we analyze the current 
    # distribution of remaining capacities in the system to determine 
    # "opportunity cost".
    
    # 1. Analyze the landscape of all bins (not just valid ones for the item)
    # We look at the remaining capacity of all bins to understand fragmentation.
    # If many bins have small remainders, we should avoid creating more small 
    # remainders (favor larger bins for small items).
    # If many bins have large remainders, we should consolidate (favor smaller 
    # bins for small items).
    
    all_remaining = bins
    n_bins = len(all_remaining)
    
    # Calculate statistics of the remaining capacity distribution
    # We use the mean and a measure of spread/fragmentation.
    # A simple proxy for fragmentation is the variance or the ratio of 
    # small remainders to total capacity.
    
    # Let's define a "fragmentation pressure" metric.
    # If the average remaining capacity is low, the system is tight.
    avg_remaining = np.mean(all_remaining)
    
    # If the system is very empty (avg_remaining high), we can afford to be 
    # more aggressive with consolidation (Best Fit).
    # If the system is tight (avg_remaining low), we should be more careful 
    # to not create unusable small slots, but actually, in tight systems, 
    # Best Fit is usually good to pack things in. 
    # However, if we have many "half-full" bins, we might want to keep 
    # "full" bins open for large items.
    
    # Let's try a different angle: 
    # Score = - (Remaining Capacity) * (1 + Penalty)
    # Where Penalty depends on how "wasteful" this bin is relative to others.
    
    # Alternative: Use the concept of "Next Fit Decreasing" adapted online.
    # But we don't know future items.
    
    # Let's try a strategy that mimics "First Fit" but with a dynamic bias.
    # First Fit tends to perform well on random sequences.
    # Best Fit tends to perform well on specific distributions.
    
    # What if we use the *rank* of the bin's remaining capacity among all bins?
    # This is invariant to the absolute scale of capacity (though here it's fixed at 100).
    
    valid_remaining = all_remaining[valid]
    
    # Calculate the rank of each valid bin's remaining capacity 
    # in the context of ALL bins.
    # Higher rank (larger remaining) -> higher score? (Worst Fit)
    # Lower rank (smaller remaining) -> higher score? (Best Fit)
    
    # Let's blend based on the item size relative to the *current average bin usage*.
    # If items are generally small, Best Fit is good.
    # If items are generally large, Worst Fit is good.
    
    # We can estimate the "typical item size" by looking at the 
    # distribution of remaining capacities.
    # If most bins have > 50 remaining, items are small.
    # If most bins have < 50 remaining, items are large.
    
    # Let's compute a dynamic threshold.
    # Median remaining capacity of all bins.
    median_remaining = np.median(all_remaining)
    
    # If the median remaining is high, the system is "young" or items are small.
    # If the median remaining is low, the system is "mature" or items are large.
    
    # Let's define a score that interpolates between Best Fit and Worst Fit 
    # based on the median remaining capacity.
    
    # Best Fit Score: -remaining
    # Worst Fit Score: remaining
    
    # If median_remaining > 50 (lots of space), use Best Fit (consolidate).
    # If median_remaining < 50 (little space), use Worst Fit (spread out)?
    # Actually, when space is little, Best Fit is often better to avoid 
    # wasting the little space left.
    
    # Let's try a different structural change: 
    # **Gap Analysis**.
    # For each valid bin, calculate the "gap" it creates if we put the item in it.
    # Gap = remaining - item.
    # We want to minimize the "regret" of this gap.
    
    # A common advanced heuristic is to look at the *density* of items.
    # Let's try a simple but structurally different approach:
    # **Simulated Annealing-like local search?** No, too slow per item.
    
    # **Greedy with Lookahead?**
    # Estimate the probability that a future item fits in the gap.
    # We don't know future items, but we can assume a distribution.
    # For Weibull, the distribution is known.
    # However, we don't have the parameters.
    
    # Let's assume the "typical" remaining capacity is the median.
    # If the gap we create is smaller than the typical item, it's "dead" space.
    # If the gap is larger, it's "live" space.
    
    # We can score based on how much "live" space we preserve.
    # Live Space = max(0, Gap - typical_item_size)
    
    # How to estimate typical_item_size?
    # We can use the standard deviation of the remaining capacities?
    # Or just a fixed heuristic?
    
    # Let's try a strategy that explicitly penalizes creating "orphan" gaps.
    # An orphan gap is one that is too small for most future items but 
    # too large to be ignored.
    
    # Let's define a "dead zone" for gaps.
    # If gap < 10, it's likely dead.
    # If gap > 20, it's likely live.
    
    # Score = - (Gap if Gap < 10 else 0) - (Gap if Gap > 20 else 0) * 0.1
    # This prefers gaps in the [10, 20] range? No, that's weird.
    
    # Let's go back to **Rank-Based Scoring**.
    # Rank the valid bins by their remaining capacity.
    # If we are in the "early" phase of packing (many bins full/empty), 
    # Best Fit is good.
    # If we are in the "late" phase (bins are partially filled), 
    # Worst Fit might be better to keep options open.
    
    # Metric for phase: Fraction of bins with remaining > 50.
    fraction_large = np.mean(all_remaining > 50)
    
    # If fraction_large is high, we are early. Use Best Fit.
    # If fraction_large is low, we are late. Use Worst Fit?
    
    # Let's test this hypothesis.
    # Early: Best Fit (minimize gap).
    # Late: Worst Fit (maximize gap to accommodate larger items that might come? 
    #        Or maybe Best Fit is still better to pack in?)
    
    # Actually, literature suggests Best Fit Decreasing is optimal offline. 
    # Online, Best Fit is generally robust.
    
    # Let's try a **Hybrid Rank Score**:
    # score = -remaining * (1 + alpha * (fraction_large - 0.5))
    
    # If fraction_large > 0.5 (early), alpha * (positive) -> higher penalty on remaining?
    # No, if we want Best Fit in early phase, we want to minimize remaining.
    # So score = -remaining.
    
    # If fraction_large < 0.5 (late), we might want to maximize remaining?
    # score = +remaining.
    
    # Let's interpolate the weight of the remaining capacity in the score.
    # weight = 2 * fraction_large - 1  # ranges from -1 to 1
    # If weight is positive (early), we penalize remaining (Best Fit).
    # If weight is negative (late), we reward remaining (Worst Fit).
    
    # score = weight * remaining
    # We want highest score.
    # If weight > 0, higher remaining -> higher score (Worst Fit). 
    # Wait, if weight > 0, score = weight * remaining. 
    # If remaining is small, score is small. If remaining is large, score is large.
    # This is Worst Fit.
    
    # I wanted Best Fit in early phase.
    # So if fraction_large > 0.5, I want Best Fit.
    # Best Fit means preferring small remaining.
    # So score should be -remaining.
    # So weight should be negative.
    
    # Let's redefine:
    # weight = 1 - 2 * fraction_large
    # If fraction_large > 0.5 (early), weight is negative.
    # score = weight * remaining = negative * remaining.
    # Small remaining -> score closer to 0 (less negative).
    # Large remaining -> score more negative.
    # So we prefer small remaining. This is Best Fit.
    
    # If fraction_large < 0.5 (late), weight is positive.
    # score = positive * remaining.
    # Large remaining -> higher score.
    # This is Worst Fit.
    
    # This seems like a plausible structural change: **Phase-Adaptive Fit**.
    
    weight = 1.0 - 2.0 * fraction_large
    
    # Add a small constant to avoid zero weight?
    # If weight is 0, all scores are 0. We need a tie-breaker.
    # Tie-breaker: Best Fit (smaller remaining is better if scores are equal).
    
    scores = weight * valid_remaining
    
    # If weight is close to 0, the scores are close to 0.
    # We need to ensure that if scores are tied, we pick the one with 
    # smaller remaining (Best Fit) as a stable default.
    # We can add a small epsilon * (-remaining) to break ties towards Best Fit.
    
    epsilon = 0.01
    scores = scores - epsilon * valid_remaining
    
    # Mask invalid bins
    scores_full = np.full_like(bins, -np.inf)
    scores_full[valid] = scores
    
    return scores_full
# EVOLVE-BLOCK-END
