"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.special

# Track start time for deadline guard
_START_TIME = time.time()
_DEADLINE_MARGIN = 15.0  # seconds before the 150s limit
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN

# Weibull distribution parameters for the generated sequences
# The task mentions "five generated Weibull sequences" with capacity 100.
# A common setup for these tasks uses Weibull(k, lambda) where items fit into bins of size 100.
# Assuming a standard Weibull with shape k=2.0 and scale lambda=25.0
# This yields a mean of ~22 and median of ~19, which is reasonable for capacity 100.
# We calculate the expected size of the next item.
# For Weibull(k, lambda), E[X] = lambda * Gamma(1 + 1/k)
_k = 2.0
_lambda_scale = 25.0
_EXPECTED_NEXT_ITEM = _lambda_scale * scipy.special.gamma(1 + 1/_k)

CAPACITY = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Lattice Gas Chemical Potential with Weibull Likelihood Derivative.
    
    We model the bin packing process as a one-dimensional lattice gas simulation.
    The score for placing an item is the negative of the local "chemical potential"
    of the remaining capacity. We approximate this chemical potential using the
    derivative of the Weibull log-likelihood with respect to the remainder size.
    
    The Weibull probability density function (PDF) for a remainder r is:
    f(r) = (k/lambda) * (r/lambda)^(k-1) * exp(-(r/lambda)^k)
    
    The log-likelihood is:
    L(r) = log(k/lambda) + (k-1)log(r/lambda) - (r/lambda)^k
    
    The derivative dL/dr (score direction) is:
    dL/dr = (k-1)/r - (k/lambda) * (r/lambda)^(k-1)
    
    We want to maximize the likelihood of the resulting remainder being "useful"
    or "stable" in the gas analogy. However, in bin packing, we generally want
    to avoid remainders that are too small (dead space) or too large (wasted capacity 
    that could have been used for other items if we had packed tighter).
    
    Actually, the "chemical potential" approach usually suggests moving towards
    states of higher probability density. But for bin packing, the "ideal" state 
    is often a remainder that is either 0 (perfect fit) or large enough to 
    accommodate future large items.
    
    Let's reinterpret: The "cost" of a remainder is the negative log-likelihood 
    or a potential derived from it. 
    
    Inspiration 1 (Best-Fit + Probabilistic) worked well. 
    Inspiration 2 (Polarization) worked well.
    
    Let's combine:
    1. Base score: -gap (Best Fit)
    2. Probabilistic Adjustment: Penalize gaps that are unlikely to be filled 
       by the next item. If the gap is much smaller than the expected next item, 
       it's likely dead space. If it's much larger, it's less efficient.
    
    The derivative of the Weibull log-likelihood with respect to r:
    dL/dr = (k-1)/r - (k * r^(k-1)) / lambda^k
    
    For k=2, lambda=25:
    dL/dr = 1/r - (2 * r) / 625
    
    This derivative is positive for small r (increasing likelihood as r increases 
    from 0) and negative for large r (decreasing likelihood as r increases).
    The peak is at r = lambda * sqrt((k-1)/k) = 25 * sqrt(0.5) ≈ 17.7.
    
    This suggests that remainders around 17.7 are the most "likely" or "natural" 
    for the Weibull distribution. 
    However, in bin packing, we want to AVOID creating remainders that are 
    "stuck" in the middle where they can't fit many future items.
    
    Let's use the score as:
    Score = - (gap * multiplier)
    
    Where multiplier is derived from the likelihood.
    If the gap is close to the mode of the Weibull distribution (17.7), it's a 
    "natural" size. But we want to avoid it? No, we want to fill bins.
    
    Let's try a different structural change:
    Use the Weibull CDF to estimate the probability that a future item fits 
    into the remainder.
    P(item <= r) = 1 - exp(-(r/lambda)^k)
    
    If we place an item, the new remainder is r_new.
    The "value" of this remainder is the expected number of future items that 
    can fit. This is roughly r_new / E[X] if we assume uniform, but with Weibull 
    it's more complex.
    
    Simpler: 
    Score = -r_new * (1 - P(next_item <= r_new))
    This penalizes large remainders that are unlikely to be filled.
    It also penalizes small remainders (linearly).
    
    Let's combine this with the polarization idea from Inspiration 2.
    Polarization: -min(r_new, CAPACITY - r_new)
    
    New Score:
    1. Calculate r_new = bins - item
    2. Calculate P_fit = 1 - exp(-(r_new/lambda)^k)
    3. Dead Space Penalty = r_new * (1 - P_fit)
    4. Polarization Penalty = min(r_new, CAPACITY - r_new)
    5. Score = -Dead Space Penalty - weight * Polarization Penalty
    
    This is a structural change from the current program which used 
    r * S(r) + alpha * r. 
    Here, (1 - P_fit) is the Survival Function S(r). So Dead Space Penalty 
    is r * S(r). This is the same as the current program's dead_space term.
    
    The current program: Score = -(r * S(r) + alpha * r)
    Inspiration 1: Score = -r * (1 + exp(-(r-E[X])/sigma)) - 0.5 * min(r, C-r)
    
    Let's try a hybrid of Inspiration 1 and the current program, but with a 
    structural tweak: Use the Weibull PDF to weight the polarization penalty.
    
    Score = -r * multiplier - weight * pdf(r) * min(r, C-r)
    
    Where multiplier = 1 + exp(-(r - E[X]) / sigma)
    
    This makes the polarization penalty stronger where the remainder is most 
    "likely" to occur naturally, which might be counter-productive.
    
    Let's go back to the "Chemical Potential" directive.
    The directive says: "negative of the local 'chemical potential' ... 
    approximated by the derivative of the Weibull log-likelihood".
    
    dL/dr = (k-1)/r - (k * r^(k-1)) / lambda^k
    
    Let's use Score = dL/dr.
    For k=2, lambda=25:
    dL/dr = 1/r - 2r/625
    
    If r is small (e.g., 1), dL/dr = 1 - 0.003 = 0.997 (High score)
    If r is medium (e.g., 20), dL/dr = 0.05 - 0.064 = -0.014 (Low score)
    If r is large (e.g., 50), dL/dr = 0.02 - 0.08 = -0.06 (Low score)
    
    This prefers small remainders. It's essentially Best Fit with a slight 
    penalty for very small remainders (if r < sqrt((k-1)/k * lambda^k)).
    Actually, for k=2, the peak is at 17.7. 
    For r < 17.7, dL/dr > 0.
    For r > 17.7, dL/dr < 0.
    
    So it prefers r=0 (limit is infinity) and avoids r=17.7.
    This seems like a good heuristic: prefer perfect fits, avoid "natural" 
    mid-sized gaps that might be dead space.
    
    Let's add a small Best-Fit term to ensure it doesn't pick a huge bin 
    if a small one is available? No, dL/dr goes to 0 as r->infinity. 
    So it prefers small r.
    
    Let's combine dL/dr with a linear Best-Fit penalty to ensure monotonic 
    decrease for large r?
    Score = dL/dr - beta * r
    
    For r=1, Score = 0.997 - beta
    For r=20, Score = -0.014 - 20*beta
    For r=50, Score = -0.06 - 50*beta
    
    If beta is small (0.01), 
    r=1: 0.987
    r=20: -0.214
    r=50: -0.56
    
    This strongly prefers small gaps.
    
    Let's try this structural change.
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
    
    # Calculate derivative of Weibull log-likelihood
    # dL/dr = (k-1)/r - (k * r^(k-1)) / lambda^k
    # Handle r=0 case (perfect fit) -> derivative is infinity. 
    # We want to assign a very high score to perfect fits.
    # np.where(r > 0, ..., large_value)
    
    k = _k
    lam = _lambda_scale
    
    # Avoid division by zero
    safe_r = np.where(r > 0, r, 1.0)
    
    term1 = (k - 1) / safe_r
    term2 = (k * (safe_r ** (k - 1))) / (lam ** k)
    
    dL_dr = term1 - term2
    
    # For r=0, dL_dr should be very high.
    # Let's set it to a large constant, e.g., 1000.
    dL_dr = np.where(r > 0, dL_dr, 1000.0)
    
    # Add a linear penalty to ensure that if dL_dr is negative for large r,
    # it becomes even more negative, reinforcing Best Fit.
    beta = 0.05
    linear_penalty = beta * r
    
    final_scores = dL_dr - linear_penalty
    
    scores[feasible] = final_scores
    
    return scores
# EVOLVE-BLOCK-END
