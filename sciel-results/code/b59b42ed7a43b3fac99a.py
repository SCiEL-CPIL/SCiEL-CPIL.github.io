"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Adversarial robustness scoring: each bin's score is the negative of the
    maximum possible future regret, where regret is defined as the size of the
    largest Weibull-distributed item that could fit into the resulting remainder.
    
    This prioritizes bins that create remainders robust against worst-case
    immediate future demand. We model future items as Weibull(1, 10) distributed
    (common for this task setup). The largest item that could fit in remainder r
    is bounded by r itself, but we weight by the probability that such a large
    item would actually arrive soon.
    
    Score = -expected_regret, where expected_regret approximates the integral
    of P(item > x) * x over x in [0, r], which simplifies to a function of r.
    For Weibull(k=1, lambda=10), the survival function is exp(-x/10).
    E[min(item, r)] = ∫₀^r P(item > t) dt = ∫₀^r exp(-t/10) dt = 10*(1 - exp(-r/10))
    
    But regret is better modeled as: if we leave remainder r, the wasted space
    is r - (next item that fits). Expected waste ≈ E[max(r - item, 0) | item ≤ r] * P(item ≤ r) + r * P(item > r)
    
    Simplified: score = -[r * exp(-r/10) + 10*(1 - exp(-r/10))] * constant
    This penalizes both very small r (can't fit typical items) and very large r 
    (wastes capacity that could have been used).
    
    Actually, let's use a simpler robustness measure: 
    score = - (r - 10*(1 - exp(-r/10)))
    This is negative of the "dead space" relative to what a typical item would use.
    
    For r=0: score = 0
    For r=10: score = -(10 - 10*(1-exp(-1))) = -(10 - 6.32) = -3.68
    For r=50: score = -(50 - 10*(1-exp(-5))) = -(50 - 9.93) = -40.07
    For r=100: score = -(100 - 10*(1-exp(-10))) = -(100 - 10) = -90
    
    Higher score (less negative) is preferred. This favors moderate remainders
    around 10-20 where typical Weibull items can still fit efficiently.
    """
    # Deadline guard: this is O(n) so should be fast, but guard just in case
    start = time.perf_counter()
    deadline = start + 1.0  # 1 second safety margin for this function
    
    # Filter bins with enough capacity
    remaining = bins - item
    valid_mask = remaining >= 0
    
    # For Weibull(k=1, lambda=10): 
    # expected_regret(r) = r - 10*(1 - exp(-r/10))
    # This is the dead space not used by a typical item
    # We want to minimize regret, so score = -regret
    
    r = np.where(valid_mask, remaining, 0)
    # Avoid numerical issues for very large r
    r_safe = np.clip(r, 0, 200)
    exp_term = np.exp(-r_safe / 10.0)
    expected_used = 10.0 * (1.0 - exp_term)
    regret = r_safe - expected_used
    
    # Score: negative regret (lower regret = higher score)
    scores = -regret
    
    # Set invalid bins to very negative score
    scores = np.where(valid_mask, scores, -1e10)
    
    elapsed = time.perf_counter() - start
    if elapsed > deadline:
        # Emergency fallback: simple best-fit
        return np.where(valid_mask, -remaining, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
