"""Improved program: Adaptive Best-Fit with Weibull-aware scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN

def _weibull_cdf(x, k=1.5, lam=1.0):
    """Compute Weibull CDF for array x."""
    x = np.asarray(x)
    cdf = np.zeros_like(x)
    mask = x > 0
    if np.any(mask):
        cdf[mask] = 1.0 - np.exp(-((x[mask] / lam) ** k))
    return cdf

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for placing the given item.
    bins: array of remaining capacities.
    Returns: array of scores, higher is better.
    Only bins with enough capacity are scored; others get -inf.
    """
    # Deadline guard
    if time.time() - _START_TIME > _MAX_EVAL_TIME:
        remaining = bins - item
        scores = np.where(remaining >= 0, -remaining, -np.inf)
        return scores

    remaining = bins
    # Mask bins that can fit the item
    mask = remaining >= item
    
    if not np.any(mask):
        return -np.inf * np.ones_like(remaining)
    
    # Adaptive strategy:
    # 1. For large items, use Best Fit (minimize remaining space).
    # 2. For small items, soften the preference to avoid fragmenting small bins
    #    or to use a Weibull-aware score that considers the probability of filling.
    
    # We use a sigmoid to blend between a "neutral" score (or slightly Worst-Fit)
    # and Best Fit.
    # Actually, the inspiration 1 suggests using -CDF(remaining) which is Best-Fit-like but non-linear.
    # Let's combine:
    # Base score: Best Fit = -(remaining - item)
    # Weibull score: -_weibull_cdf(remaining - item)
    
    # Let's stick to the current program's structure:
    # coeff * remaining + offset?
    # Current: coeff = -sigmoid(item - threshold). scores = coeff * remaining.
    # This makes scores = -remaining for large items (Best Fit) and 0 for small items (Neutral/First Fit-ish if ties).
    # If small items get score 0, the evaluator picks the first one? Or arbitrary?
    # Usually, if scores are tied, the index or some default is used.
    # To be safe, we should always have a preference order.
    
    # Let's refine the "small item" behavior.
    # Instead of neutral (0), maybe use a very weak Best Fit or a Weibull-based score.
    # Weibull CDF of the *remaining space after placement*?
    # If we place item in bin with remaining R, new remaining is R - item.
    # We want this new remaining to be "good".
    # If new remaining is small, it's likely to be wasted (if no future items fit).
    # If new remaining is large, it might be filled.
    # Actually, for small items, R - item is still large.
    
    # Let's try a hybrid score:
    # score = -alpha * (remaining - item) - beta * CDF(remaining - item)
    # This penalizes both absolute waste and the "likelihood" of that waste being unavoidable.
    
    # However, the current program's "neutral" for small items (coeff -> 0) effectively becomes "First Fit" (or arbitrary).
    # First Fit is often worse than Best Fit.
    # So we should probably keep a Best Fit bias even for small items, but perhaps weaker?
    # Or, use the Weibull CDF score which is always Best-Fit-like but non-linear.
    
    # Let's test the Weibull CDF score from Inspiration 1.
    # Score = -CDF(remaining - item).
    # Since CDF is increasing, this minimizes (remaining - item).
    # So it's Best Fit.
    # But it's non-linear. For large gaps, the score is close to -1. For small gaps, it's close to 0.
    # This compresses the differences between large gaps, making the choice less sensitive to large gaps?
    # No, if gap=1, CDF~0.05, score~-0.05.
    # If gap=10, CDF~0.9, score~-0.9.
    # It still prefers gap=1.
    
    # The key difference is the *relative* weighting.
    
    # Let's try to combine the adaptive coefficient with the Weibull score.
    # For large items: Strong Best Fit.
    # For small items: Weibull Best Fit.
    
    # Actually, let's look at the "neutral" issue.
    # If coeff is 0, scores are 0.
    # If all scores are 0, which bin is chosen?
    # The problem says "goes to the highest-scoring bin". If tied, it's undefined or first.
    # To avoid undefined behavior, we should ensure scores are distinct or have a clear preference.
    
    # Let's modify the small-item strategy to be a *weak* Best Fit instead of Neutral.
    # coeff = -sigmoid(item - threshold)
    # For small items, sigmoid ~ 0, coeff ~ 0.
    # Let's shift the sigmoid so that small items still get a small negative coefficient.
    # Or, add a small constant Best Fit term.
    
    # score = -alpha * remaining
    # alpha = 0.1 + 0.9 * sigmoid(item - threshold)
    # For small items, alpha = 0.1 (Weak Best Fit).
    # For large items, alpha = 1.0 (Strong Best Fit).
    
    threshold = 2.0
    k = 1.0
    sig = 1.0 / (1.0 + np.exp(-k * (item - threshold)))
    alpha = 0.1 + 0.9 * sig
    
    scores = -alpha * remaining
    
    # Mask out bins that don't fit.
    scores[~mask] = -np.inf
    
    return scores

# EVOLVE-BLOCK-END
