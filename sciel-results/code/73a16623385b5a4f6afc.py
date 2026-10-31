"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.special

# Deadline guard: keep a module-level start time to enforce the 150s limit.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill

# Weibull parameters for estimating fill probability of remainders
# Tuned for typical Weibull sequences with capacity 100
_WEIBULL_K = 2.0
_WEIBULL_LAMBDA = 25.0  # Scale parameter

# Expected size of the next item for Weibull(k, lambda)
# E[X] = lambda * Gamma(1 + 1/k)
_EXPECTED_NEXT_ITEM = _WEIBULL_LAMBDA * scipy.special.gamma(1 + 1.0 / _WEIBULL_K)

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Returns an array of scores (higher is better for selection).
    
    Strategy: Probabilistic Dead-Space Minimization with Fragmentation Penalty.
    
    This combines two structural ideas:
    1. From the current program: A "Dead Space" metric based on the Weibull survival
       function. DeadSpace(r) = r * S(r) where S(r) = exp(-(r/lambda)^k).
       This identifies a "dead zone" of medium-sized remainders that are worst for
       future packing.
    2. From Inspiration 1: A fragmentation penalty that strongly discourages remainders
       that are too small to likely fit the next item (i.e., r < E[X]).
    
    The score is computed as:
    Score = - (DeadSpace(r) + FragmentationPenalty(r))
    
    Where:
    - DeadSpace(r) = r * exp(-(r/lambda)^k) captures the bimodal nature of remainder utility.
    - FragmentationPenalty(r) is a smooth step that penalizes r when r < E[X].
      We use: Penalty = E[X] * exp(-(r - E[X]) / sigma) for r < E[X], else 0.
      Actually, a smoother version: Penalty = E[X] * sigmoid((E[X] - r) / sigma)
      Or simply: if r < E[X], we add a large penalty that decays as r approaches E[X].
      
    Let's refine the fragmentation penalty to be smooth and differentiable-like:
    Let threshold = _EXPECTED_NEXT_ITEM.
    Let sigma = 5.0.
    Penalty(r) = threshold * exp(-(r - threshold) / sigma) if r < threshold, else 0.
    This creates a strong repulsion from the [0, threshold) region.
    
    Combined with DeadSpace, which is low near 0 and high in the middle,
    the fragmentation penalty ensures that even though DeadSpace is low near 0,
    we don't pick a bin that leaves a tiny fragment (e.g., r=1) if a bin with
    r=15 (which is > E[X] ~ 22? No, E[X] for k=2, lambda=25 is 25*Gamma(1.5) ~ 22)
    is available. Wait, if E[X] is 22, then r=15 is penalized. r=1 is heavily penalized.
    r=30 is not penalized by fragmentation, and DeadSpace(30) is low (since 30 > r*).
    
    Let's check the values:
    k=2, lambda=25.
    r* (max dead space) = lambda / sqrt(k) = 25 / 1.414 ~ 17.6.
    E[X] = 25 * 0.886 = 22.15.
    
    So the dead zone is around r=17.6.
    The fragmentation threshold is around r=22.15.
    
    If we pick r=10:
    DeadSpace(10) = 10 * exp(-(10/25)^2) = 10 * exp(-0.16) = 10 * 0.852 = 8.52.
    Fragmentation(10) = 22.15 * exp(-(10-22.15)/5) = 22.15 * exp(2.43) = 22.15 * 11.3 = 250.
    Total Penalty ~ 258. Score = -258.
    
    If we pick r=20:
    DeadSpace(20) = 20 * exp(-(20/25)^2) = 20 * exp(-0.64) = 20 * 0.527 = 10.54.
    Fragmentation(20) = 22.15 * exp(-(20-22.15)/5) = 22.15 * exp(0.43) = 22.15 * 1.53 = 33.9.
    Total Penalty ~ 44.4. Score = -44.4.
    
    If we pick r=30:
    DeadSpace(30) = 30 * exp(-(30/25)^2) = 30 * exp(-1.44) = 30 * 0.237 = 7.11.
    Fragmentation(30) = 0 (since 30 > 22.15).
    Total Penalty ~ 7.11. Score = -7.11.
    
    If we pick r=50:
    DeadSpace(50) = 50 * exp(-(50/25)^2) = 50 * exp(-4) = 50 * 0.018 = 0.9.
    Fragmentation(50) = 0.
    Total Penalty ~ 0.9. Score = -0.9.
    
    This seems to strongly prefer large remainders (Worst Fit) when they are above the fragmentation threshold.
    Is that good?
    For bin packing, Worst Fit is generally worse than Best Fit for total bins used.
    However, the "Dead Space" model suggests that large remainders are "safe" because they can fit future items.
    But if we always pick the largest remainder, we spread items out and waste capacity.
    
    Let's re-evaluate the "Dead Space" hypothesis.
    The original code had a `beta * r` term to prevent pure Worst Fit.
    Let's add that back.
    
    Score = - (DeadSpace(r) + FragmentationPenalty(r) + beta * r)
    
    With beta = 0.1:
    r=30: Penalty = 7.11 + 0 + 3.0 = 10.11. Score = -10.11.
    r=50: Penalty = 0.9 + 0 + 5.0 = 5.9. Score = -5.9.
    r=20: Penalty = 10.54 + 33.9 + 2.0 = 46.44. Score = -46.44.
    r=10: Penalty = 8.52 + 250 + 1.0 = 259.5. Score = -259.5.
    
    It still prefers r=50 over r=30.
    Maybe beta needs to be larger, or the fragmentation penalty needs to be tuned differently.
    
    Let's look at Inspiration 1 again. It used:
    Score = -gap * (1 + exp(-(gap - E[X]) / sigma))
    This is a multiplicative penalty on the gap itself.
    
    Let's try a hybrid:
    Base Score = -r (Best Fit)
    Modifier = 1 + FragmentationFactor(r)
    Where FragmentationFactor(r) is large when r < E[X].
    
    Actually, let's stick to the additive penalty structure but tune the parameters.
    The key insight from the current program (score 1.817) was that reducing beta helped.
    But 1.817 is already good. We want to improve it.
    
    Let's try a different structural change:
    Instead of a fixed Weibull parameter, can we infer the distribution?
    No, we only have `bins` and `item`.
    
    Let's try the "Polarization" idea from Inspiration 2 but combined with the Weibull insight.
    Inspiration 2 scored by distance from C/2. This is a simple heuristic.
    
    Let's try a "Fit Quality" score based on how well the item fits the bin relative to the expected next item.
    
    New Strategy:
    1. Calculate remaining `r`.
    2. If `r` is very small (< 5), it's a fragment. Penalize heavily.
    3. If `r` is medium (5 to 30), it's in the danger zone. Score based on Weibull Dead Space.
    4. If `r` is large (> 30), it's a "safe" large gap. Score based on Best Fit (prefer smaller r) but less penalized than medium.
    
    Let's implement a piecewise or smooth function that achieves this.
    
    Score(r) = - ( alpha * r + beta * DeadSpace(r) + gamma * Fragmentation(r) )
    
    Let's tune alpha, beta, gamma.
    alpha = 0.5 (Best Fit bias)
    beta = 1.0 (Dead Space bias)
    gamma = 10.0 (Fragmentation bias)
    
    Let's test mentally:
    r=1: alpha*1 + beta*DeadSpace(1) + gamma*Frag(1)
    DeadSpace(1) ~ 1 * 1 = 1.
    Frag(1) ~ high.
    Score is very low.
    
    r=10: alpha*10 + beta*8.5 + gamma*Frag(10).
    Frag(10) is high.
    
    r=20: alpha*20 + beta*10.5 + gamma*Frag(20).
    Frag(20) is moderate.
    
    r=30: alpha*30 + beta*7.1 + gamma*0.
    Score = - (15 + 7.1) = -22.1.
    
    r=50: alpha*50 + beta*0.9 + gamma*0.
    Score = - (25 + 0.9) = -25.9.
    
    r=15: alpha*15 + beta*DeadSpace(15) + gamma*Frag(15).
    DeadSpace(15) = 15 * exp(-(15/25)^2) = 15 * exp(-0.36) = 15 * 0.697 = 10.4.
    Frag(15) = 22 * exp(-(15-22)/5) = 22 * exp(1.4) = 22 * 4.05 = 89.
    Score = - (7.5 + 10.4 + 89) = -106.9.
    
    This seems to prefer r=30 over r=50, which is good (Best Fit).
    And it strongly avoids r=15 (dead zone + fragmentation).
    
    Let's refine the Fragmentation function.
    Frag(r) = _EXPECTED_NEXT_ITEM * exp(-(r - _EXPECTED_NEXT_ITEM) / sigma) if r < _EXPECTED_NEXT_ITEM else 0.
    This is discontinuous at r = _EXPECTED_NEXT_ITEM.
    Let's make it smooth:
    Frag(r) = _EXPECTED_NEXT_ITEM * sigmoid((_EXPECTED_NEXT_ITEM - r) / sigma)
    sigmoid(x) = 1 / (1 + exp(-x))
    
    At r = E[X], sigmoid(0) = 0.5. Frag = 0.5 * E[X] ~ 11.
    At r = 0, sigmoid(inf) = 1. Frag = E[X] ~ 22.
    At r = E[X] + 10, sigmoid(-2) ~ 0.12. Frag ~ 2.7.
    
    This adds a constant penalty of ~11 around the threshold.
    
    Let's try this combination.
    
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: return best-fit scores quickly
        remaining = bins - item
        mask = remaining >= 0
        scores = np.full(len(bins), -np.inf)
        if np.any(mask):
            scores[mask] = -remaining[mask]
        return scores

    # Calculate remaining capacity
    remaining = bins - item
    
    # Identify valid bins (enough capacity)
    valid = remaining >= 0
    
    # Initialize scores with -inf for invalid bins
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid):
        return scores
        
    valid_rems = remaining[valid]
    r = np.maximum(valid_rems, 0.0)
    
    # 1. Dead Space Metric: r * exp(-(r/lambda)^k)
    ratio = r / _WEIBULL_LAMBDA
    power_term = np.power(ratio, _WEIBULL_K)
    survival = np.exp(-power_term)
    dead_space = r * survival
    
    # 2. Fragmentation Penalty: Smooth step based on expected next item
    sigma = 5.0
    # Use sigmoid to create a smooth transition
    # x = (E[X] - r) / sigma
    x = (_EXPECTED_NEXT_ITEM - r) / sigma
    # Clip x to avoid overflow in exp
    x_clipped = np.clip(x, -10, 10)
    sigmoid_val = 1.0 / (1.0 + np.exp(-x_clipped))
    fragmentation = _EXPECTED_NEXT_ITEM * sigmoid_val
    
    # 3. Best Fit Bias: Linear penalty on remainder
    alpha = 0.5
    best_fit_penalty = alpha * r
    
    # Combine penalties
    total_penalty = dead_space + fragmentation + best_fit_penalty
    
    # Score is negative penalty (higher is better)
    scores[valid] = -total_penalty
    
    return scores
# EVOLVE-BLOCK-END
