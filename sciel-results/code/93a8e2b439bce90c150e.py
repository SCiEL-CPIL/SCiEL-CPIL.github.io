"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard state
_START_TIME = None
_DEADLINE = 330.0  # Wall clock limit
_SAFETY_MARGIN = 5.0  # Seconds before limit to stop

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_exceeded():
    global _START_TIME, _DEADLINE, _SAFETY_MARGIN
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)

def acquisition(mu, sigma, f_best, cost):
    _init_time()
    
    # Ensure inputs are numpy arrays for speed
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety check: if we are running out of time, return a simple fallback
    if _time_exceeded():
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)

    n = len(mu)
    
    # 1. Compute base Expected Improvement for each point
    # EI(x) = E[max(0, f_best - f(x))]
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. Estimate Remaining Budget
    # The problem states: "Runs start with 5 uniform random points... While total cost is below 30"
    # We don't know the exact total cost spent so far, but we can estimate it.
    # The `cost` array contains the cost of the *candidates*.
    # The total cost spent is roughly 5 * (1 + mean(uniform)) + sum(costs of evaluated points).
    # However, we can approximate the "remaining budget" for the *current* decision step.
    # Since we are picking ONE point to evaluate next, the relevant constraint is that 
    # the cost of this point must be affordable.
    # But Batch EI logic suggests we should look ahead.
    
    # Let's use a "Greedy Batch Selection" approach to compute a "Batch-Adjusted EI".
    # Instead of just EI/cost, we simulate adding this point to a hypothetical batch 
    # of size B (determined by remaining budget) and see how much *additional* improvement 
    # it provides compared to the best single point.
    
    # To keep it fast and vectorized, we approximate the "Marginal Value of Information" 
    # by combining EI with a term that rewards points that are "diverse" from the current best 
    # in the cost-value space, or simply use a stronger cost penalty that mimics 
    # the efficiency of packing more points.
    
    # Structural Change: 
    # Instead of EI / cost, we use a "Cost-Effective Improvement" metric that 
    # accounts for the *probability* that this point leads to a new best, weighted by 
    # the *expected reduction* in the search space uncertainty (a proxy for future efficiency).
    
    # A more direct structural change: **Thompson Sampling with Cost Constraint**
    # Sample f(x) ~ N(mu, sigma^2).
    # Score = 1 if f(x) < f_best else 0, but weighted by 1/cost?
    # No, that's stochastic. We need a deterministic score for the evaluator.
    
    # Let's try **Expected Hypervolume Improvement (EHI)** approximation.
    # Hypervolume in (cost, value) space.
    # Current reference point: (f_best, current_total_cost_estimate).
    # This is hard without knowing current_total_cost.
    
    # Let's go back to the finding: 1/cost^2 was better.
    # Why? Because it allows more points.
    # If we assume the GP is decent, the *value* of a point is its EI.
    # The *cost* of a point is `cost`.
    # The *efficiency* is EI / cost.
    # The *total efficiency* of a sequence of points is sum(EI_i / cost_i) only if they are independent.
    # But they are not.
    
    # New Strategy: **Multi-Step Lookahead via Greedy Batch Construction**
    # We will compute a score that approximates the EI of a point *if it were part of a batch* 
    # of size k, where k is estimated from the remaining budget.
    # 
    # Approximation:
    # The marginal gain of adding point i to a set S is EI(i) - max_{j in S} EI(j) * correlation(i,j).
    # Since we don't have correlations, we assume that points with *different costs* are less correlated 
    # in their "value for money" potential.
    # 
    # Simpler Structural Change: **Cost-Adaptive UCB**
    # UCB = mu + beta * sigma.
    # We want to minimize. So we look for low mu + high sigma?
    # Actually, for minimization, UCB is mu - beta * sigma (lower bound of confidence interval).
    # But we want to be optimistic about finding a low value.
    # So we use the Lower Confidence Bound (LCB) for minimization?
    # LCB = mu - beta * sigma.
    # Points with low mu and high sigma get very low LCBs (optimistic).
    # 
    # Score = -LCB / cost = (beta * sigma - mu) / cost.
    # This is structurally different from EI. EI is non-linear and saturates.
    # LCB is linear.
    # 
    # Let's test LCB with a cost penalty.
    # To match the "more points is better" insight, we use a strong cost penalty.
    
    # Hyperparameter: beta.
    # In standard BO, beta is often ~ 1-5.
    # Since we have standardized targets (implied by GP setup), sigma is in objective units.
    # Let's pick beta = 2.0.
    
    beta = 2.0
    lcb = mu - beta * sigma
    
    # We want to maximize the score.
    # Low lcb is good.
    # So score = -lcb / cost^gamma.
    
    # The finding said cost^2 was better than cost^1.
    # Let's use cost^2.
    
    cost_safe = np.maximum(cost, 1.0)
    
    score = -lcb / (cost_safe ** 2)
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # One more check: LCB can be very negative if sigma is large.
    # This encourages exploration.
    # EI encourages exploitation when mu < f_best.
    # LCB encourages exploration even when mu > f_best if sigma is large enough.
    # This might help escape local optima.
    
    return score

# EVOLVE-BLOCK-END
