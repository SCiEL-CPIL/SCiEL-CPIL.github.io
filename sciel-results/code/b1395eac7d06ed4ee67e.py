"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state for budget tracking
_start_time = None
_total_cost_budget = 30.0

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    
    # Initialize start time on first call
    if _start_time is None:
        _start_time = time.time()
    
    # 1. Time Guard: Ensure we return quickly to allow the main loop to proceed
    #    The task states "Each evaluation must finish within 300 s".
    #    Our computation is vectorized and fast, but we add a safety check.
    elapsed = time.time() - _start_time
    if elapsed > 280:
        # Fallback to simple EI if we are dangerously close to timeout
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-6)
    
    # 2. GP-UCB with budget-scaled exploration
    #    UCB for minimization: we want to maximize the "upper confidence bound"
    #    of improvement, which is equivalent to minimizing mu - beta * sigma
    #    where beta controls exploration vs exploitation.
    
    # Estimate how much budget has been used based on elapsed time and typical cost per iteration
    # The loop starts with 5 points (cost ~7.5), then evaluates one point per iteration.
    # Each iteration involves scoring 2000 candidates + evaluating 1 point.
    # We can estimate the number of iterations completed from elapsed time.
    # However, a simpler approach: use the mean cost of candidates as a proxy for
    # "how expensive the search is", which correlates with budget consumption.
    
    # Better: use a fixed schedule based on the fact that we have ~30-7.5 = 22.5 budget
    # and each evaluation costs ~1.5 on average, so ~15 evaluations total.
    # We'll use a linear decay from high exploration to low exploitation.
    
    # Use the median cost of candidates as a rough indicator of where in the domain
    # we are searching, but for budget tracking, we'll use a heuristic based on
    # the number of times this function has been called (approximated by elapsed time).
    
    # Simple heuristic: assume each acquisition call takes ~0.1s, so we can estimate
    # the number of iterations. But this is fragile.
    
    # Alternative: use a fixed exploration coefficient that balances well for
    # the expected number of evaluations (~15).
    
    # Let's use a UCB with beta that decays linearly from 2.0 to 0.5 over the budget.
    # We estimate progress as (elapsed / 300) * 0.8 to stay within bounds.
    progress = min(1.0, elapsed / 250.0)  # Scale to finish decay before timeout
    beta = 2.0 - 1.5 * progress  # Decays from 2.0 to 0.5
    
    # Compute UCB score for minimization:
    # We want to minimize f, so we want low mu and high sigma (uncertainty).
    # UCB for minimization: score = -mu + beta * sigma
    # This gives higher scores to points with low predicted value and high uncertainty.
    ucb_scores = -mu + beta * sigma
    
    # 3. Cost-aware penalty using convex combination
    #    The cost function is 1 + mean(x_scaled), ranging from 1 to 2.
    #    We want to penalize expensive points, but not too aggressively.
    #    Use a convex combination: penalty = alpha * cost + (1 - alpha) * 1.0
    #    where alpha controls the strength of cost awareness.
    
    # From the insight that reducing cost penalty helps, we use a mild penalty.
    # Let's use: score = ucb_scores / (0.5 + 0.5 * cost)
    # This normalizes cost to [0.5, 1.0] when cost is in [1, 2].
    
    # Actually, let's use a softer penalty:
    # score = ucb_scores - lambda * (cost - 1.5)
    # where lambda is small, so we don't heavily penalize.
    
    # Or: score = ucb_scores / (1.0 + 0.1 * (cost - 1.0))
    # This gives a penalty that ranges from 1.0 (cost=1) to 1.1 (cost=2).
    
    # Let's use: score = ucb_scores / (1.0 + 0.2 * (cost - 1.0))
    # This gives a penalty from 1.0 to 1.2.
    
    cost_penalty = 1.0 + 0.2 * (cost - 1.0)
    scores = ucb_scores / cost_penalty
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
