"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Weibull parameters estimated for the specific distribution regime
# Capacity 100, 1000 items. Typical Weibull for such packing problems
# often has shape k ~ 2.0 and scale lambda ~ 15-20.
# We use k=2.0 and lambda=16.0 as a robust baseline.
K = 2.0
LAMBDA = 16.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score is preferred.
    
    Strategy:
    - Estimate the expected "waste" or "dead space" created by placing the item.
    - Let r be the remaining capacity after placing the item.
    - The next item X is drawn from Weibull(K, LAMBDA).
    - If X > r, the next item cannot fit, and the space r becomes "dead" (wasted) 
      until a much smaller item arrives, or the bin is closed.
    - If X <= r, the next item fits. The waste is not immediate, but we incur 
      the cost of leaving (r - X) space.
    - We approximate the immediate penalty as the probability that the next item 
      does NOT fit, weighted by the gap size.
    - Specifically, we want to minimize E[Waste].
    - A simple proxy for "badness" of gap r is:
      1. If r is very small, it's good (bin is full).
      2. If r is large, it's risky because the next item might be small, leaving 
         a large gap, or might be large, not fitting.
    
    Let's refine the score:
    Score = - (r * S(r) + (1 - S(r)) * E[X | X <= r] * 0) 
    Actually, a better heuristic for "efficiency" is:
    We prefer bins where the remaining space r is "compatible" with the next item.
    
    Let's use the expected "utilization" of the remaining space by the next item.
    If we place item in bin with remaining r:
    - Prob next item fits: CDF(r)
    - If it fits, we use min(X, r) of the space. Since X <= r, we use X.
    - If it doesn't fit (X > r), we use 0 of the space (it's wasted for this step).
    
    Expected space used by next item from this gap:
    E[Use] = ∫_0^r x * f(x) dx + 0 * ∫_r^∞ f(x) dx
    This integral is related to the partial expectation.
    For Weibull, E[X * 1(X<=r)] = ∫_0^r x * (k/λ)(x/λ)^(k-1) exp(-(x/λ)^k) dx
    
    Alternatively, simpler heuristic:
    Score = - (r - E[X | X <= r] * P(X <= r)) ? No.
    
    Let's look at Inspiration 2: "negative of the gap *after* subtracting the expected size of the next item".
    Score = - (r - E[X_next]) where E[X_next] is the expected size of the next item, 
    but capped or modified by the probability it fits?
    
    Actually, Inspiration 2 says: "score for each bin is the negative of the gap *after* subtracting the expected size of the next item".
    This implies: Score = - (r - E[X]).
    This is constant for all bins if E[X] is constant! That can't be right.
    It must mean: Score = - (r - E[X | X fits in r]) * P(X fits) ?
    Or Score = - (r - min(r, E[X]))?
    
    Let's try a "Fit Quality" score:
    We want to maximize the probability that the next item fits AND uses a significant portion of the space.
    
    Let's stick to a robust probabilistic waste model:
    Waste(r) = r * S(r) + (1 - S(r)) * (r - E[X | X <= r])
    Wait, if X fits, the waste is r - X.
    If X doesn't fit, the waste is r (for this step).
    
    Expected Waste = S(r) * r + (1 - S(r)) * E[r - X | X <= r]
    = r * S(r) + (1 - S(r)) * (r - E[X | X <= r])
    = r * S(r) + r * (1 - S(r)) - (1 - S(r)) * E[X | X <= r]
    = r - (1 - S(r)) * E[X | X <= r]
    = r - E[X * 1(X <= r)]
    
    So, Minimizing Expected Waste is equivalent to Maximizing E[X * 1(X <= r)].
    E[X * 1(X <= r)] = ∫_0^r x f(x) dx.
    
    This is a smooth, monotonically increasing function of r?
    Let's check. 
    If r=0, integral is 0.
    If r->inf, integral is E[X].
    So E[X * 1(X <= r)] increases with r.
    This means we prefer LARGER r? That would be Worst-Fit.
    But we established Worst-Fit is often bad.
    
    Why? Because E[Waste] = r - E[X * 1(X <= r)].
    If r is huge, E[X * 1(X <= r)] approaches E[X].
    So Waste approaches r - E[X].
    If r is small, E[X * 1(X <= r)] is small.
    Waste approaches r.
    
    Let's compare r=10 vs r=50.
    If E[X]=15.
    r=10: Waste = 10 - E[X * 1(X<=10)]. Since most X are >10, E[X*1(X<=10)] is small (maybe 2). Waste ~ 8.
    r=50: Waste = 50 - E[X * 1(X<=50)]. Most X are <50. E[X*1(X<=50)] ~ 14. Waste ~ 36.
    
    So minimizing expected immediate waste favors SMALL r (Best-Fit).
    
    So the strategy is: Score = - (r - E[X * 1(X <= r)])
    Or simply: Score = E[X * 1(X <= r)] - r
    
    Let's implement this.
    E[X * 1(X <= r)] for Weibull(k, lambda):
    f(x) = (k/λ) (x/λ)^(k-1) exp(-(x/λ)^k)
    Integral I(r) = ∫_0^r x (k/λ) (x/λ)^(k-1) exp(-(x/λ)^k) dx
    
    Let u = (x/λ)^k => x = λ u^(1/k), dx = (λ/k) u^(1/k - 1) du
    Limits: 0 to (r/λ)^k
    
    I(r) = ∫_0^(r/λ)^k λ u^(1/k) * (k/λ) * u^((k-1)/k) * exp(-u) * (λ/k) u^(1/k - 1) du
    Wait, x f(x) dx:
    x = λ u^(1/k)
    f(x) dx = (k/λ) (x/λ)^(k-1) exp(-u) dx
    dx = (λ/k) u^(1/k - 1) du
    (x/λ)^(k-1) = u^((k-1)/k)
    
    So x f(x) dx = λ u^(1/k) * (k/λ) u^((k-1)/k) exp(-u) * (λ/k) u^(1/k - 1) du
    = λ * (k/λ) * (λ/k) * u^(1/k + (k-1)/k + 1/k - 1) * exp(-u) du
    = λ * u^(1/k + 1 - 1/k) * exp(-u) du  -- Wait.
    1/k + (k-1)/k = 1/k + 1 - 1/k = 1.
    So u^1 * u^(1/k - 1) = u^(1/k).
    
    Let's re-calculate powers:
    x = λ u^(1/k)
    f(x) = (k/λ) u^((k-1)/k) exp(-u)
    dx = (λ/k) u^(1/k - 1) du
    
    Product:
    λ u^(1/k) * (k/λ) u^((k-1)/k) * (λ/k) u^(1/k - 1) * exp(-u)
    Coefficients: λ * (k/λ) * (λ/k) = λ
    Powers of u: 1/k + (k-1)/k + 1/k - 1 = 1/k + 1 - 1/k + 1/k - 1 = 1/k.
    
    So I(r) = λ ∫_0^(r/λ)^k u^(1/k) exp(-u) du
    
    This is λ * Γ(1 + 1/k, (r/λ)^k) ... wait, lower incomplete gamma is ∫_0^z t^{s-1} e^{-t} dt.
    Here s-1 = 1/k => s = 1 + 1/k.
    So I(r) = λ * γ(1 + 1/k, (r/λ)^k)
    
    Where γ(s, z) is the lower incomplete gamma function.
    scipy.special.gammaincc? No, gammainc is regularized lower incomplete gamma P(s,z) = γ(s,z)/Γ(s).
    So γ(s,z) = Γ(s) * P(s,z).
    
    I(r) = λ * Γ(1 + 1/k) * scipy.special.gammainc(1 + 1/k, (r/λ)^k)
    
    We can precompute Γ(1 + 1/k).
    
    Score = I(r) - r
    
    This score is:
    - For small r: I(r) ~ r^(1+1/k) ... derivative at 0 is 0?
      I(r) ≈ λ * ( (r/λ)^k * (1/k+1)^{-1} )? No.
      For small z, γ(s,z) ≈ z^s / s.
      I(r) ≈ λ * [ (r/λ)^k * (1+1/k)^{-1} ] / (1+1/k) ?
      Let's just use the formula.
      
    If r is very small, I(r) is very small (order r^(1+1/k) if k>0? No, r^k * r^(1/k) = r^(k+1/k)? No.
    Integral of x^(k) roughly.
    Actually, for small r, I(r) is small. Score = I(r) - r ≈ -r.
    So for small r, Score is negative and increases as r decreases (less negative).
    Wait, if r=0, Score=0. If r=1, Score ≈ -1.
    So we prefer r=0 over r=1. (Best-Fit behavior for small gaps).
    
    If r is large, I(r) -> λ Γ(1+1/k) = E[X].
    Score = E[X] - r.
    Since r is large, Score is very negative.
    So we prefer smaller r over large r.
    
    This confirms the score favors Best-Fit (small r).
    But does it favor "too small" r?
    If r=0.1, Score ≈ -0.1.
    If r=10, Score = I(10) - 10.
    If E[X]=15, I(10) might be 8. Score = 8 - 10 = -2.
    So r=0.1 (Score -0.1) is better than r=10 (Score -2).
    
    This seems to strongly favor Best-Fit.
    Is that good?
    Best-Fit is generally good.
    
    Let's add a time guard.
    """
    scores = np.full(len(bins), -1e18)
    
    remaining = bins - item
    valid_mask = remaining >= 0
    if not np.any(valid_mask):
        return scores
    
    valid_indices = np.where(valid_mask)[0]
    valid_r = remaining[valid_mask]
    
    # Precompute constant
    # gamma_1_plus_1_over_k = math.gamma(1 + 1.0/K)
    # We can import math or use scipy
    try:
        from scipy.special import gammainc, gamma
        gamma_s = gamma(1.0 + 1.0/K)
    except:
        # Fallback if scipy not available (though task says it is)
        import math
        gamma_s = math.gamma(1.0 + 1.0/K)
    
    # Calculate (r/lambda)^k
    z = np.power(valid_r / LAMBDA, K)
    
    # Calculate lower incomplete gamma ratio P(s, z)
    # gammainc(s, z) returns P(s, z)
    p_vals = gammainc(1.0 + 1.0/K, z)
    
    # I(r) = lambda * gamma(s) * P(s, z)
    expected_used = LAMBDA * gamma_s * p_vals
    
    # Score = Expected Used - r
    # We want to maximize this.
    scores[valid_indices] = expected_used - valid_r
    
    return scores
# EVOLVE-BLOCK-END
