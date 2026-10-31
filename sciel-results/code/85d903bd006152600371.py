"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Weibull parameters inferred from typical bin packing benchmarks
# Assuming shape (k) ~ 2 and scale (lambda) ~ 12-15 for items in [0, 100]
# These are estimates to guide the lookahead simulation.
WEIBULL_K = 2.0
WEIBULL_LAMBDA = 12.0

# Pre-compute a lookup table for the Weibull CDF and PDF to speed up evaluation
# if we were doing many samples, but here we use analytical moments/expectations.
# For the lookahead, we estimate the expected "waste" after placing item i and item i+1.

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin within 120s limit
    
    # Determine which bins have enough remaining capacity
    valid = bins >= item
    
    if not np.any(valid):
        return np.full_like(bins, -np.inf)
        
    # Compute remaining capacity after placing the item
    remaining = bins - item
    
    # Initialize scores. We want to minimize waste.
    # Score will be negative of estimated future waste.
    
    # Strategy: Lookahead 1 step.
    # For each bin with remaining space R:
    # 1. Place current item. New space = R.
    # 2. Estimate what happens with the next item (X ~ Weibull).
    #    - If X <= R, it fits. New space = R - X.
    #    - If X > R, it doesn't fit. New space = R.
    # We want to minimize the expected "dead space" or maximize the "utilization".
    
    # A common heuristic for online bin packing is to avoid "fragmentation".
    # Fragmentation occurs when R is large enough to not be negligible, but too small to fit most future items.
    
    # Let's approximate the expected remaining space after the next item.
    # E[Space_after] = P(X > R) * R + E[(R - X) | X <= R] * P(X <= R)
    # Note: (R - X) is positive only if X <= R.
    # E[(R - X) * 1(X <= R)] = integral_0^R (R - x) f(x) dx
    #                         = R * F(R) - integral_0^R x f(x) dx
    # Where F is CDF and f is PDF.
    
    # Weibull CDF: F(x) = 1 - exp(-(x/lambda)^k)
    # Weibull PDF: f(x) = (k/lambda) * (x/lambda)^(k-1) * exp(-(x/lambda)^k)
    
    # To keep it fast and robust, we can use a simpler proxy for "goodness":
    # A gap R is "good" if:
    # 1. R is very small (bin is nearly full).
    # 2. R is large enough that the probability of the next item fitting is high, 
    #    AND the expected remaining space after fitting is still manageable.
    
    # Let's define a "Waste Score" for a given remaining space R.
    # Waste(R) = R * (1 - P_fit(R)) + Expected_Residual_Waste_if_Fits(R)
    
    # Actually, a better metric for bin packing is the "Best Fit Decreasing" variant logic.
    # Let's try a score that penalizes gaps that are "in the middle".
    # We define a function w(r) that is high for r near 0 and r near lambda*k, 
    # and low for r in between? No, we want to MINIMIZE waste.
    
    # Let's use the expected remaining space after the next item as the cost.
    # Cost(R) = E[Space_after_next]
    # We want to minimize this cost. So Score = -Cost(R).
    
    # However, if R is very large, Cost(R) is also large (because R - X is large).
    # This would prefer small bins (Best Fit).
    # But Best Fit can lead to many small bins being opened if items are large.
    # We need to balance this.
    
    # Alternative: Score based on "Utilization Potential".
    # Let's stick to the "Dead Zone" penalty but refined with lookahead.
    
    # Calculate P_fit = F(R)
    rem_clipped = np.maximum(remaining, 0.0)
    x = rem_clipped / WEIBULL_LAMBDA
    F_R = 1.0 - np.exp(-np.power(x, WEIBULL_K))
    
    # Calculate E[(R - X) | X <= R] * P(X <= R) = R * F(R) - E[X * 1(X<=R)]
    # E[X * 1(X<=R)] = integral_0^R x f(x) dx
    # For Weibull, this integral doesn't have a simple closed form for arbitrary k, 
    # but we can approximate or use the fact that for k=2, it's related to the Gamma function.
    # For k=2: integral_0^R x * (1/lambda^2) x exp(-x^2/lambda^2) dx
    # Let u = x^2/lambda^2, du = 2x/lambda^2 dx => x dx = lambda^2/2 du
    # Integral = 1/2 lambda^2 * integral_0^{R^2/lambda^2} exp(-u) du
    #          = 1/2 lambda^2 * (1 - exp(-R^2/lambda^2))
    #          = 1/2 lambda^2 * F(R)
    
    # So for k=2:
    # E[X * 1(X<=R)] = 0.5 * lambda^2 * F(R)
    # E[(R-X) * 1(X<=R)] = R * F(R) - 0.5 * lambda^2 * F(R) = F(R) * (R - 0.5 * lambda^2)
    
    # Total Expected Remaining Space after next item:
    # E[Space_after] = P(X > R) * R + E[(R - X) | X <= R] * P(X <= R)
    #                = (1 - F(R)) * R + F(R) * (R - 0.5 * lambda^2)
    #                = R - R*F(R) + R*F(R) - 0.5 * lambda^2 * F(R)
    #                = R - 0.5 * lambda^2 * F(R)
    
    # This is a very clean formula for k=2!
    # Cost(R) = R - 0.5 * lambda^2 * (1 - exp(-(R/lambda)^2))
    
    # We want to minimize Cost(R).
    # Score = -Cost(R) = -(R - 0.5 * lambda^2 * F(R))
    
    # Let's check the behavior:
    # If R is small: F(R) ~ (R/lambda)^k. Cost ~ R. Score ~ -R. (Prefers smaller R, i.e., Best Fit)
    # If R is large: F(R) ~ 1. Cost ~ R - 0.5 lambda^2. Score ~ -R + const. (Still prefers smaller R)
    # This looks like pure Best Fit with a slight bias.
    
    # However, Best Fit is known to be suboptimal for some distributions because it fills bins too tightly, 
    # leaving no room for variation.
    # Let's look at the derivative of Cost(R) with respect to R.
    # d/dR [R - 0.5 lambda^2 F(R)] = 1 - 0.5 lambda^2 f(R)
    # If 1 - 0.5 lambda^2 f(R) > 0, Cost is increasing, so we prefer smaller R.
    # If 1 - 0.5 lambda^2 f(R) < 0, Cost is decreasing, so we prefer larger R.
    
    # f(R) = (k/lambda) (R/lambda)^(k-1) exp(-(R/lambda)^k)
    # For k=2, f(R) = (2R/lambda^2) exp(-(R/lambda)^2)
    # 0.5 lambda^2 f(R) = R exp(-(R/lambda)^2)
    # Condition: 1 - R exp(-(R/lambda)^2) < 0 => R exp(-(R/lambda)^2) > 1
    # Max of x exp(-x^2) is at x=1/sqrt(2), value is 1/sqrt(2e) ~ 0.42.
    # So R exp(-(R/lambda)^2) is always < 1.
    # Thus Cost(R) is always increasing.
    # This implies that minimizing expected remaining space is equivalent to Best Fit.
    
    # So the "Lookahead 1 step" with expected space minimization reduces to Best Fit.
    # We need a different objective.
    
    # Objective: Minimize the number of bins.
    # This is equivalent to maximizing the "density" of used space.
    # Or, avoiding the creation of "unfillable" gaps.
    
    # Let's try a different metric: The probability that the bin becomes "full" (or nearly full) 
    # after the next item.
    # P(Full_after) = P(X >= R - epsilon) for some epsilon?
    # No, that's not quite right.
    
    # Let's look at Inspiration 1: "effective capacity".
    # Score = - (R - E[X | X <= R] * F(R))? No.
    
    # Let's try a hybrid: 
    # We want to avoid gaps that are too small to be useful (dead) and too large to be efficient.
    # Let's define a "Utility" function U(R).
    # U(R) is high if R is close to 0 (bin filled) or if R is large enough to fit a large item.
    
    # Actually, a known strong heuristic is "Worst Fit" for some distributions and "Best Fit" for others.
    # For Weibull with k=2 (symmetric-ish, heavy tail), Best Fit is usually good.
    
    # Let's try to incorporate the *variance* of the next item.
    # If the next item is highly variable, we might want to keep a larger buffer.
    
    # New Strategy: **Stochastic Best Fit**
    # Score = - (R + Penalty(R))
    # Where Penalty(R) accounts for the risk that R is too small to fit the next item, 
    # leading to wasted space in the current bin and forced opening of a new bin.
    
    # If R < X_next, we waste R.
    # If R >= X_next, we use R - X_next.
    
    # Let's penalize R if it is likely to be wasted.
    # Waste_Prob(R) = P(X > R) = 1 - F(R)
    # Expected_Waste(R) = R * (1 - F(R)) + E[R - X | X <= R] F(R)
    # This is the same as Cost(R) above.
    
    # Since Cost(R) is monotonic, Best Fit is the optimal policy for minimizing expected remaining space.
    # But we are minimizing *bins*, not *remaining space*.
    # Minimizing remaining space is a proxy.
    
    # Is there a case where Best Fit is bad?
    # Yes, if items are large and bins are small, Best Fit might fill many bins partially.
    
    # Let's try **First Fit Decreasing** logic? No, items arrive online.
    
    # Let's try a **Gap-Size Distribution** approach.
    # We want the remaining gap R to be drawn from a distribution that matches the item distribution.
    # If R is distributed like X, then P(R >= X) is maximized?
    
    # Let's try a score that encourages R to be either 0 or > lambda.
    # Score = -R * (1 + alpha * (1 - F(R)))
    # This is similar to the seed program.
    
    # Let's try a **Simulated Annealing** style local search? No, too slow.
    
    # Let's try **Lookahead 2 steps**.
    # This is computationally expensive but might be within budget if vectorized.
    # For each bin, simulate 1000 next items? No, too slow.
    # Analytical 2-step lookahead?
    
    # Let's go back to the "Dead Zone" concept.
    # The seed program uses: scores = -remaining * (2.0 - prob_fit)
    # This penalizes small gaps (prob_fit=0) with factor 2, and large gaps (prob_fit=1) with factor 1.
    # This is a weighted Best Fit.
    
    # My proposal: Use a **Quadratic Penalty** for gaps in the "middle".
    # Define a target gap size G_target = 0.
    # But we know G=0 is ideal.
    # Let's define a penalty for "fragmentation".
    # Fragmentation is high when R is small but not zero, and next item doesn't fit.
    
    # Let's try: Score = -R - beta * R * (1 - F(R))
    # This is the seed.
    
    # What if we invert the logic?
    # Score = F(R) - R/lambda
    # Maximize probability of next item fitting, penalize by size.
    
    # Let's try a **Neural Network** approximation? No, no training data.
    
    # Let's try a **Combinatorial** heuristic:
    # Check if the bin can fit the current item AND the median of the next item?
    # If R >= item + median_next, it's a "safe" bin.
    # If R < item + median_next, it's a "risky" bin.
    
    # Median of Weibull(k=2, lambda=12) = lambda * (ln 2)^(1/k) = 12 * 0.8326 = 9.99.
    # So if R >= item + 10, it's safe.
    
    # Let's score bins based on "Safety".
    # Safe bins (R > item + 10) get a bonus.
    # Unsafe bins get a penalty.
    
    # This is a structural change: Binary classification of bins into "Safe" and "Unsafe" with a tie-breaker.
    
    # Let's refine:
    # 1. Calculate R = bins - item.
    # 2. Calculate R_safe_threshold = item + WEIBULL_LAMBDA * (ln 2) ** (1/WEIBULL_K)
    # 3. For bins with R >= R_safe_threshold:
    #    Score = -R (Best Fit among safe bins)
    # 4. For bins with R < R_safe_threshold:
    #    Score = -R - penalty (Penalize unsafe bins)
    
    # This encourages filling safe bins first, then unsafe ones.
    # This is different from the continuous penalty in the seed.
    
    # Let's implement this.
    
    # Calculate median
    median_next = WEIBULL_LAMBDA * (np.log(2)) ** (1.0 / WEIBULL_K)
    
    # Threshold for "Safe" bins: Remaining space after current item should be at least median_next
    # to have a 50% chance of fitting the next item.
    # Actually, if R >= median_next, P(X <= R) >= 0.5.
    
    # We want to prioritize bins that are "full" (small R) but only among those that are "safe"?
    # No, if a bin is unsafe, it will likely waste space.
    
    # Let's try:
    # Score = -R if R >= median_next else -R - 10.0
    
    # This creates a "step" in the scoring function.
    
    # Let's verify the logic.
    # If we have two bins:
    # Bin A: R = 5 (Unsafe, < 10)
    # Bin B: R = 15 (Safe, > 10)
    # Seed:
    # A: prob_fit ~ 0.2. Score = -5 * (2 - 0.2) = -9
    # B: prob_fit ~ 0.8. Score = -15 * (2 - 0.8) = -18
    # Seed prefers A (Best Fit).
    
    # My Proposal:
    # A: Score = -5 - 10 = -15
    # B: Score = -15
    # Tie.
    
    # If I increase penalty to 20:
    # A: -25
    # B: -15
    # Proposal prefers B (Worst Fit among safe, or rather, avoids unsafe).
    
    # This is a structural shift from "Minimize Immediate Waste" to "Avoid Future Waste".
    
    # Let's tune the penalty.
    # Penalty should be proportional to the expected waste if it fails.
    # Expected waste if R < X is R.
    # So Penalty = R * (1 - F(R))?
    # That brings us back to the seed.
    
    # Let's try a **Fixed Penalty** for being below the median.
    
    scores = -remaining.copy()
    
    # Identify unsafe bins
    unsafe = remaining < median_next
    
    # Apply penalty to unsafe bins
    # The penalty should discourage placing items in bins that will likely have leftover space
    # that cannot be used by the next item.
    # A fixed penalty of 5.0 (half the median) seems reasonable.
    penalty = 5.0
    scores[unsafe] -= penalty
    
    # Mask invalid bins
    scores[~valid] = -np.inf
    
    return scores

# EVOLVE-BLOCK-END
