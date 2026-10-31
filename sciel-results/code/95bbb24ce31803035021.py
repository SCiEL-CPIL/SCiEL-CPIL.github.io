import numpy as np
import time

# Global state
_start_time = None

# Weibull distribution parameters for item sizes
# k=2.0 (shape), lambda=12.0 (scale)
# This gives mean ~10.6, with a reasonable tail
_WEIBULL_K = 2.0
_WEIBULL_LAMBDA = 12.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    global _start_time
    
    # Initialize timing if not started
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: keep 5s margin before 150s limit
    elapsed = time.time() - _start_time
    if elapsed > 145:
        # Fallback to simple Best Fit
        remaining = bins - item
        valid = remaining >= 0
        scores = np.full(len(bins), -np.inf)
        if np.any(valid):
            scores[valid] = -remaining[valid]
        return scores

    # Calculate remaining capacity
    remaining = bins - item
    
    # Identify valid bins (enough capacity)
    valid = remaining >= 0
    
    # Initialize scores with -inf for invalid bins
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid):
        return scores
        
    valid_indices = np.where(valid)[0]
    new_rems = remaining[valid]
    
    # Distribution-aware scoring:
    # We want to minimize expected dead space (wasted capacity) in the remainder.
    # 
    # Key insight: For a remainder r, the "waste" depends on whether future items
    # can fit. We use the survival function S(r) = P(X > r) as a proxy for
    # "how likely this space is to be useless."
    #
    # But we also need to balance consolidation. A purely Best-Fit approach
    # (minimize r) works well, but a distribution-aware approach can do better
    # by recognizing that some remainders are "dead" (too small to be useful)
    # while others are "alive" (likely to fit future items).
    #
    # Strategy: Use a composite score that penalizes:
    # 1. Large remainders that are unlikely to fit future items (dead space)
    # 2. Very small remainders (wastes a bin that could be consolidated)
    #
    # The function f(r) = r * S(r) where S(r) = exp(-(r/lambda)^k) has:
    # - f(0) = 0 (small remainder, but S(0)=1, so product is small)
    # - f(r) peaks at r* = lambda * k^(-1/k) ~ 8.5 for our parameters
    # - f(r) -> 0 as r -> infinity (large remainder, but S(r) -> 0)
    #
    # So f(r) is low at both extremes and high in the middle.
    # We want to AVOID the middle (dead zone) and prefer either very small
    # or very large remainders.
    #
    # But for bin packing, we generally prefer consolidation (smaller r).
    # So we use: score = - (f(r) + beta * r)
    # where beta is a small consolidation bias.
    
    # Compute survival function: S(r) = exp(-(r/lambda)^k)
    scaled = new_rems / _WEIBULL_LAMBDA
    survival = np.exp(-np.power(scaled, _WEIBULL_K))
    
    # Expected size of next item that fits in remainder r:
    # E[X | X <= r] = r - ∫_0^r S(t) dt
    # For Weibull, ∫_0^r S(t) dt = λ * Γ(1+1/k) * γ(k, (r/λ)^k) / k
    # where γ is the lower incomplete gamma. We approximate with a simpler form:
    # effective_waste(r) = r * S(r) + λ/k * (1 - S(r))
    # This captures: dead space (r*S(r)) plus the "lost opportunity" of space
    # that could have fit smaller items.
    
    # Dead space: r * S(r) — high when remainder is medium-sized
    dead_space = new_rems * survival
    
    # Opportunity cost: space that's too small for the "typical" next item
    # but not zero. Use (1 - S(r)) as proxy for "fraction of items that fit"
    # The wasted potential is roughly proportional to λ * (1 - S(r)) when r < λ
    # We scale this down to avoid dominating
    opportunity_cost = 0.3 * _WEIBULL_LAMBDA * (1.0 - survival)
    
    # Consolidation bias: prefer slightly smaller remainders to reduce fragmentation
    consolidation = 0.03 * new_rems
    
    # Total penalty to minimize
    penalty = dead_space + opportunity_cost + consolidation
    
    # Score is negative penalty (higher is better)
    scores_valid = -penalty
    
    # Small tie-breaker: prefer lower index for stability
    scores_valid -= np.arange(len(valid_indices)) * 1e-10
    
    scores[valid] = scores_valid
    
    return scores
