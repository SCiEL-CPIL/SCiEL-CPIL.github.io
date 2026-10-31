"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline guard
_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0  # seconds before the 150s limit
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN

# Weibull distribution parameters for the generated sequences
# Based on high-performing inspiration, use k=2.0, lambda=15.0
# Mean item size ~ 15 * Gamma(1.5) ≈ 13.3
_k = 2.0
_lambda_scale = 15.0
CAPACITY = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Harmonic Gap Utilization with Consolidation Bias.
    
    Instead of probabilistic dead-space modeling, we use a structural approach:
    1. Primary objective: Maximize the "usefulness" of the remaining gap.
       A gap r is useful if it can fit a significant number of future items.
       We approximate this by the harmonic mean of the gap size and the capacity,
       which favors moderate gaps that can fit multiple items but are not too large
       to be wasteful.
       
       Actually, let's use a simpler and more robust metric:
       Score = - (r^2 / (r + item))
       
       This penalizes large gaps (quadratically) but is normalized by the current item size.
       If item is small, large gaps are heavily penalized. If item is large, the penalty
       is less severe, encouraging consolidation.
       
       Wait, a better structural insight: Best Fit Decreasing (BFD) is near-optimal.
       The online equivalent is Best Fit (BF): place in the bin with the smallest remaining
       gap that fits the item. This minimizes wasted space per item.
       
       However, pure BF can be suboptimal when it leaves many small gaps that can't fit
       anything. A hybrid approach:
       Score = - (remaining_gap * weight)
       where weight increases as the gap becomes "unusable".
       
       Let's use the "Next Fit Decreasing" insight: if we sort items, small items go last.
       Online, we don't know the future. But we know the distribution.
       
       New Structural Approach: Gap Efficiency
       For a gap r, the efficiency is how well it can be filled.
       If r is very small (< min_item), it's dead.
       If r is large, it has potential.
       
       Let's use: Score = - (1 / (1 + r)) * r  = - r / (1 + r)
       This is bounded in [-1, 0]. It favors smaller gaps (Best Fit) but saturates.
       
       Actually, let's look at the inspiration score 1.8171. The current is 1.35.
       The parent had lambda=15.0. The current has lambda=25.0.
       The change from 15 to 25 hurt the score significantly (1.8 -> 1.35).
       So lambda=15.0 is likely better.
       
       Let's revert to the parent's parameters and structure but simplify.
       The parent likely used a simpler scoring function.
       
       Let's try a pure Best Fit variant with a slight consolidation boost.
       Score = - remaining_gap
       This is pure Best Fit.
       
       To improve on Best Fit, we can penalize gaps that are "too large" relative to
       the expected item size, as they are likely to be partially wasted.
       
       Let's use: Score = - (remaining_gap + lambda * exp(-remaining_gap / lambda))
       This adds a penalty for large gaps that decays as gaps get larger? No, exp(-r/lambda)
       decays as r increases. So for large r, the penalty is just r. For small r, the penalty
       is r + lambda. This actually favors larger gaps? No, we want to minimize penalty.
       So we want to minimize r + lambda * exp(-r/lambda).
       Derivative: 1 - exp(-r/lambda) = 0 -> r=0. So minimum at r=0.
       This is just Best Fit with an offset.
       
       Let's try: Score = - (r * (1 - exp(-r / lambda)))
       For small r, r*(1-r/lambda) ≈ r. For large r, r*(1-0) = r.
       This is almost linear.
       
       Let's go back to the parent's likely approach.
       The parent score was 1.8171. The current is 1.3516.
       The difference is in the Weibull parameters and the scoring formula.
       
       Let's use the parent's parameters (k=2, lambda=15) and a scoring function that
       combines Best Fit with a penalty for "dead" space.
       
       Dead space for a gap r is approximately the expected wasted space if we fill it
       with random items. For Weibull(k, lambda), the expected waste in a bin of size r
       is complex.
       
       Simple effective heuristic:
       Score = - (r + c * (r / lambda)^2)
       This penalizes large gaps quadratically.
       
       Let's try: Score = - (r + 0.1 * (r / 15.0)**2)
    """
    # Check if we're close to the deadline
    if time.time() - _START_TIME > _MAX_EVAL_TIME:
        # Emergency: return a simple best-fit fallback
        remaining = bins - item
        scores = np.where(remaining >= 0, -remaining, -np.inf)
        return scores
    
    remaining_after = bins - item
    feasible = remaining_after >= 0
    
    if not np.any(feasible):
        # No feasible bins; return all -inf
        return np.full(len(bins), -np.inf)
    
    scores = np.full(len(bins), -np.inf)
    
    # Get feasible gaps
    r = remaining_after[feasible]
    
    # Structural scoring:
    # Primary: Best Fit (minimize r)
    # Secondary: Penalize gaps that are significantly larger than the item,
    #            as they may lead to fragmentation.
    # We use a weighted sum: penalty = r + w * (r - item) * exp(-(r - item)/scale)
    # where (r - item) is the "extra" space left.
    
    # Simpler: Just use Best Fit with a slight preference for bins that are
    # "fuller" (i.e., have larger item sizes already? We don't have that info).
    # We only have 'bins' which are the remaining capacities.
    
    # Let's use the harmonic mean of the gap and a reference size.
    # Reference size = expected item size ~ 13.3
    # Score = - (r / (r + 13.3))
    # This is bounded and favors smaller r.
    
    ref_size = 13.3
    penalty = r / (r + ref_size)
    
    scores[feasible] = -penalty
    
    return scores
# EVOLVE-BLOCK-END
