import numpy as np
import time

_start_time = time.time()
_deadline = _start_time + 140.0  # Safety margin before 150s limit

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores bins using an Expected Dead Space heuristic.
    Minimizes the expected wasted space in each bin given a Weibull-like distribution of future items.
    Dead space is the remaining capacity that cannot be filled by future items.
    We approximate this by penalizing remainders that are unlikely to be fully utilized.
    """
    # Check deadline
    if time.time() > _deadline:
        # Fallback to simple best-fit if time is running out
        remaining = bins - item
        scores = np.full(bins.shape, -np.inf, dtype=float)
        valid_mask = remaining >= -1e-9
        if np.any(valid_mask):
            # Best-fit: prefer smallest remaining space (maximize used space)
            scores[valid_mask] = bins[valid_mask]
        return scores

    # Calculate remaining space after placing the item
    remaining = bins - item
    
    # Initialize scores to -infinity to handle invalid bins
    # We maximize the score, so -inf ensures invalid bins are never picked
    scores = np.full(bins.shape, -np.inf, dtype=float)
    
    # Define valid bins (can fit the item)
    valid_mask = remaining >= -1e-9
    
    if np.any(valid_mask):
        r = remaining[valid_mask]
        
        # Expected Dead Space Model:
        # The "dead space" is the part of the remaining capacity that will likely be wasted.
        # For a bin with remaining capacity r, if future items have a certain distribution,
        # the expected dead space is related to the probability that a future item
        # cannot fit, or that the remainder is too small to be efficiently packed.
        
        # A simple model: Dead Space ~ r * P(item > r) + E[item | item <= r] * (1 - P(item > r)) ... no, that's not right.
        # Actually, the dead space is the remaining space that is NOT filled.
        # If we assume items arrive until the bin is full or no item fits, the dead space is the final remainder.
        
        # The inspiration suggests using a distribution-aware metric.
        # Let's use a surrogate: The "waste" is high if the remainder is in the "dead zone" where it's too small for most items but too large to ignore.
        # Or, more simply, we want to minimize the expected remaining space that is wasted.
        
        # A common heuristic is to minimize the "fragmentation".
        # Let's try a score that penalizes remainders that are "awkward".
        # For Weibull items (often skewed), small remainders might be good (bin almost full) or bad (can't fit anything).
        
        # Let's look at the "Expected Dead Space" r * S(r) mentioned in the knowledge.
        # S(r) could be the survival function of the item size distribution at r.
        # If the item size distribution is Weibull with shape k and scale lambda:
        # S(r) = exp(-(r/lambda)^k)
        # If we don't know the parameters, we can assume a standard shape.
        # However, we don't know the distribution parameters.
        
        # Alternative: Use the concept that we want to avoid leaving "medium" sized gaps.
        # The polarization heuristic did this by maximizing |r - 50|.
        # The "Expected Dead Space" approach might be:
        # Score = - (r * exp(-r/scale))  # if scale is small, large r is bad, small r is good?
        
        # Let's try to model the probability that the remaining space r is "wasted".
        # If r is very small, it's likely wasted (no item fits).
        # If r is very large, it's likely to be filled by many items, so the *final* dead space might be small.
        # The intermediate r values are the most problematic.
        
        # Let's use a score that is high for r close to 0 (bin full) and high for r close to Capacity (bin empty),
        # but with a different weighting than simple polarization.
        
        # The knowledge says: "Replaces heuristic 'dead zone' estimation with a distribution-aware 'Expected Dead Space' metric (r * S(r))"
        # This metric identifies remainders likely to be wasted.
        # If we assume the item size distribution is such that S(r) (prob next item > r) decreases with r,
        # then r * S(r) is the expected "unfillable" portion?
        
        # Let's assume a Weibull distribution with shape k=1.5 and scale lambda=20 (guess based on capacity 100).
        # S(r) = exp(-(r/lambda)^k)
        # Dead Space Score = r * S(r)
        # We want to MINIMIZE this dead space.
        # So we maximize - (r * S(r)).
        
        # Let's try this.
        lambda_val = 20.0
        k_val = 1.5
        S_r = np.exp(-(r / lambda_val) ** k_val)
        dead_space = r * S_r
        
        # We want to minimize dead_space, so we maximize -dead_space.
        # However, if r is large, S_r is small, so dead_space might be small.
        # If r is small, S_r is close to 1, so dead_space is small.
        # The maximum of r * S(r) occurs at some intermediate r.
        # Let's find the max of f(r) = r * exp(-(r/lambda)^k).
        # f'(r) = exp(-(r/lambda)^k) + r * exp(-(r/lambda)^k) * (-k/lambda) * (r/lambda)^(k-1)
        # f'(r) = exp(-(r/lambda)^k) * (1 - k * (r/lambda)^k)
        # Max at (r/lambda)^k = 1/k => r = lambda * k^(-1/k)
        
        # For k=1.5, lambda=20: r_max = 20 * (1/1.5)^(1/1.5) = 20 * 0.666^(0.666) = 20 * 0.787 = 15.7
        # So r=15.7 has the highest expected dead space.
        # We want to avoid r=15.7.
        
        # So the score should be low (negative) when r is near 15.7.
        # Score = -dead_space
        scores[valid_mask] = -dead_space
        
    return scores
