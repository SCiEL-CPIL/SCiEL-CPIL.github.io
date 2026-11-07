"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_ACQ_START_TIME = time.time()
_ACQ_DEADLINE = _ACQ_START_TIME + 280
_ACQ_CALL_COUNT = [0]


def acquisition(mu, np_sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of rosenbrock 2d.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions
    - np_sigma: array of shape (2000,), GP standard deviation predictions
    - f_best: scalar, best observed value so far
    - cost: array of shape (2000,), evaluation costs for each candidate
    
    Returns:
    - array of shape (2000,), acquisition scores (higher = better)
    """
    _ACQ_CALL_COUNT[0] += 1

    # Deadline guard
    if time.time() > _ACQ_DEADLINE:
        # Fallback: simple cost-aware mean
        fallback = -mu / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(fallback), fallback, 0.0)

    # Sanitize
    sigma = np.maximum(np_sigma, 1e-12)
    cost = np.maximum(cost, 1e-9)

    # Cost-Aware UCB Strategy:
    # We want to minimize the function. The "upper confidence bound" of the function value
    # is mu + beta * sigma. We want this to be as low as possible.
    # However, we also care about cost.
    # A common cost-aware approach is to maximize (f_best - (mu + beta * sigma)) / cost
    # or simply minimize (mu + beta * sigma) / cost.
    
    # Since we are minimizing, a low value of (mu + beta * sigma) is good.
    # Dividing by cost means we prefer low cost.
    # But we want to return a score where HIGHER is better.
    # So we should maximize: (f_best - (mu + beta * sigma)) / cost?
    # Or maybe: - (mu + beta * sigma) / cost?
    
    # Let's look at the inspiration: "Switching from non-linear Expected Improvement to linear Cost-Aware UCB changes the exploration/exploitation trade-off... Measured: 1 -> 1.724."
    # The diff in the observation shows:
    # "Cost-Aware UCB: score = (f_best - mu + beta * sigma) / cost"
    # Wait, that formula in the comment was cut off.
    # Standard UCB for minimization: minimize mu + beta * sigma.
    # Acquisition score (higher is better) should be negative of that, or relative improvement.
    
    # Let's use: score = (f_best - mu - beta * sigma) / cost
    # This rewards points that are likely to be lower than f_best (f_best - mu > 0)
    # and adds a penalty for uncertainty (beta * sigma) if we want to be conservative?
    # No, UCB explores by ADDING beta * sigma to the mean for minimization (making the bound higher/worse) to ensure we don't miss a low value hidden by noise?
    # Actually, for minimization, the "Upper Confidence Bound" on the *true* function value is mu + beta * sigma.
    # If we believe the true value could be as high as mu + beta * sigma, then the *improvement* over f_best is at least f_best - (mu + beta * sigma).
    # If this is negative, it's not promising.
    # So, Expected Improvement-like score: max(0, f_best - (mu + beta * sigma)) / cost.
    # But linear UCB usually just takes the bound.
    
    # Let's try: score = - (mu + beta * sigma) / cost
    # This directly minimizes the cost-normalized upper bound.
    # Higher score = lower cost-normalized upper bound.
    
    # Adaptive beta:
    # Early on, we want to explore more (higher beta).
    # Later, we want to exploit (lower beta).
    n_calls = _ACQ_CALL_COUNT[0]
    
    # Decay beta from 1.0 to 0.1 over 50 calls
    # beta = 0.1 + 0.9 * exp(-n_calls / 50.0)
    # Let's use a slightly different schedule.
    # The cost varies between 1 and 2.
    # mu and sigma are in objective units.
    
    # Let's normalize mu and sigma to make beta dimensionless-ish?
    # Or just tune beta.
    
    # If f_best is close to 0, mu is small.
    # Let's try beta = 0.5 initially.
    
    # A safer linear approach:
    # score = (f_best - mu) / cost - beta * sigma / cost
    # This is equivalent to (f_best - (mu + beta * sigma)) / cost.
    
    # Let's use a fixed beta that works well for the scale.
    # The standard deviation sigma might be large initially.
    
    # Let's try:
    # score = (f_best - mu - beta * sigma) / cost
    
    # We need to ensure this is finite.
    
    # Let's experiment with beta.
    # If beta is too high, we explore too much.
    # If beta is too low, we exploit too much.
    
    # Given the inspiration score of 1.724, let's try to replicate a similar linear UCB.
    # The comment in the diff said: "score = (f_best - mu + beta * sigma) / cost"
    # Wait, if we want to minimize mu, and we add beta*sigma, we are making the value larger (worse).
    # If we subtract beta*sigma, we are making the value smaller (better).
    # For minimization, the "optimistic" estimate is mu - beta * sigma.
    # The "pessimistic" estimate (upper bound on value) is mu + beta * sigma.
    # UCB typically uses the pessimistic bound to guarantee exploration.
    # So we minimize mu + beta * sigma.
    # Score = - (mu + beta * sigma) / cost.
    # Or Score = (f_best - (mu + beta * sigma)) / cost.
    
    # Let's use Score = (f_best - mu - beta * sigma) / cost.
    # This is high when mu + beta * sigma is low relative to f_best.
    
    # Let's pick beta.
    # If f_best is ~10, mu ~10, sigma ~1.
    # beta=0.5: score = (10 - 10 - 0.5)/1 = -0.5.
    # If mu=9, sigma=1. score = (10-9-0.5)/1 = 0.5.
    # If mu=10, sigma=0. score = (10-10-0)/1 = 0.
    
    # Let's try an adaptive beta.
    # Start with higher exploration.
    beta = 0.5 + 0.5 * np.exp(-n_calls / 20.0)
    
    score = (f_best - mu - beta * sigma) / cost
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
