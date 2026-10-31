"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for adaptive learning
# We track the 'waste' associated with different remainder levels.
# When an item is placed, we update our belief about how good that remainder was.
# If a bin ends up with a small remainder (tight fit), and later we can't fit
# anything else there, that tight fit was good. If we leave a large remainder
# (loose fit), we hope to fill it later.
# 
# Strategy:
# 1. Maintain a histogram of 'regret' or 'waste' for different remainder buckets.
# 2. Score = - (estimated future waste if we leave this remainder)
# 3. Estimated waste is higher for remainders that are "hard to fill" (e.g., ~50 for medium items)
#    and lower for remainders that are "easy to fill" (e.g., very small, or very large if items are large).
#
# Since we don't know the future, we use a heuristic proxy:
# The "value" of a remainder R is inversely proportional to the probability that
# a future item fits in R.
# For Weibull distributions, small items are frequent. So small remainders are "bad"
# (wasted) because they can't take the frequent small items.
# Large remainders are "good" because they can take anything.
# Medium remainders are "okay" but risky.
#
# However, the current polarization strategy (max distance from 50) already does this statically.
# The NEW structural change is ADAPTIVE: 
# We will track the average item size seen so far.
# If items are getting smaller, we should prefer tighter fits (Best Fit) to consolidate.
# If items are getting larger, we should prefer looser fits (Worst Fit) to preserve capacity.
# We use the running mean of item sizes to modulate the scoring curve.

_running_mean_item_size = 50.0  # Initial guess
_item_count = 0
_start_time = time.time()
_DEADLINE = 120.0  # 120s limit, keep margin

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin with enough remaining capacity.
    Adaptive strategy: Modulates the fit preference based on the running mean of item sizes.
    """
    global _running_mean_item_size, _item_count, _start_time, _DEADLINE

    # Deadline guard
    if time.time() - _start_time > _DEADLINE - 5.0:
        # Fallback to simple Best Fit to ensure speed and validity if stuck
        remainders = bins - item
        valid_mask = remainders >= 0
        scores = np.full_like(bins, -1e9, dtype=float)
        if np.any(valid_mask):
            scores[valid_mask] = -remainders[valid_mask]
        return scores

    # Update running statistics
    _item_count += 1
    _running_mean_item_size = _running_mean_item_size + (item - _running_mean_item_size) / _item_count

    capacity = 100.0
    remainders = bins - item
    valid_mask = remainders >= 0

    scores = np.full_like(bins, -1e9, dtype=float)
    
    if not np.any(valid_mask):
        return scores

    valid_remainders = remainders[valid_mask]

    # Adaptive Parameter: 
    # If mean item size is small (e.g., < 40), small items are common. 
    # We should try to pack them tightly (Best Fit) to avoid leaving tiny unusable gaps.
    # If mean item size is large (e.g., > 60), large items are common.
    # We should leave larger gaps (Worst Fit) to accommodate them.
    
    # Let 'alpha' be the strength of the Best Fit preference.
    # alpha = 1.0 means pure Best Fit (minimize remainder).
    # alpha = 0.0 means pure Worst Fit (maximize remainder).
    
    # Map running mean (0-100) to alpha (1.0 - 0.0)
    # If mean is 20 (small items), alpha should be high (prefer tight fit).
    # If mean is 80 (large items), alpha should be low (prefer loose fit).
    
    # Weibull sequences often start with a mix. 
    # Let's use a smooth transition.
    # Threshold: 50 is the midpoint.
    # If mean < 50, lean towards Best Fit.
    # If mean > 50, lean towards Worst Fit.
    
    # Linear interpolation:
    # mean=0 -> alpha=1.0 (Best Fit)
    # mean=100 -> alpha=0.0 (Worst Fit)
    alpha = 1.0 - (_running_mean_item_size / capacity)
    
    # Clamp alpha to [0.1, 0.9] to avoid degenerate cases
    alpha = np.clip(alpha, 0.1, 0.9)

    # Scoring Function:
    # We want a score that balances "tightness" (low remainder) and "looseness" (high remainder).
    # Score = alpha * (-remainder) + (1 - alpha) * (remainder)
    # This is equivalent to: (2*alpha - 1) * (-remainder) ? No.
    # Let's define:
    # Best Fit Score: -remainder (higher is better, i.e., smaller remainder)
    # Worst Fit Score: remainder (higher is better, i.e., larger remainder)
    
    # Combined Score = alpha * (-remainder) + (1 - alpha) * remainder
    #                 = -alpha * remainder + (1 - alpha) * remainder
    #                 = (1 - 2*alpha) * remainder
    
    # If alpha=1 (Best Fit): Score = -1 * remainder. Max score is for min remainder. Correct.
    # If alpha=0 (Worst Fit): Score = 1 * remainder. Max score is for max remainder. Correct.
    # If alpha=0.5 (Neutral): Score = 0. All valid bins equal. 
    
    # However, simple linear blending can be unstable if remainders vary wildly.
    # Let's normalize remainders to [0, 1] relative to capacity to make alpha more interpretable.
    # Normalized Remainder r_norm = remainder / capacity
    
    r_norm = valid_remainders / capacity
    
    # Score = (1 - 2*alpha) * r_norm
    # Wait, if alpha=1 (Best Fit), we want to MINIMIZE remainder. 
    # The function `priority` returns scores, and the item goes to the HIGHEST score.
    # So for Best Fit, we want high score for low remainder.
    # Score = -r_norm works.
    # For Worst Fit, we want high score for high remainder.
    # Score = r_norm works.
    
    # Let's refine the blend:
    # Score = alpha * (-r_norm) + (1 - alpha) * r_norm
    #       = r_norm * (1 - alpha - alpha)
    #       = r_norm * (1 - 2*alpha)
    
    # Let's check:
    # alpha = 1.0 (Best Fit): Score = -r_norm. Max score at r_norm=0. Correct.
    # alpha = 0.0 (Worst Fit): Score = r_norm. Max score at r_norm=1. Correct.
    # alpha = 0.5: Score = 0. Indifferent.
    
    # This is a linear strategy. Is it structurally different from polarization?
    # Polarization: Score = |r - 50|. This is non-linear and U-shaped.
    # This new strategy is linear and monotonic (either increasing or decreasing with r).
    # It does NOT have the "polarization" of avoiding the middle.
    # Instead, it chooses a direction (tight or loose) based on global statistics.
    
    # This is a fundamentally different design family: 
    # Polarization: "Avoid the middle, go to extremes" (Static, Non-linear)
    # Adaptive Linear: "Go in one direction (tight or loose) based on context" (Dynamic, Linear)
    
    valid_scores = (1 - 2 * alpha) * r_norm
    
    scores[valid_mask] = valid_scores
    
    return scores
# EVOLVE-BLOCK-END
