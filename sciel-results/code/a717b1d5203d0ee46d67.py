"""Rewritten cost-aware acquisition for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) with adaptive exploration.
    
    Strategy:
    Instead of EI/cost, we use LCB/cost. LCB = mu - beta * sigma.
    Since we are minimizing, we want to minimize mu - beta*sigma.
    The acquisition score is (mu - beta*sigma) / cost.
    Wait, the task says "higher = evaluate next".
    So we want to maximize the score.
    If we minimize f(x), LCB is mu - beta*sigma.
    To maximize the score for the best candidate, we can use:
    score = -(mu - beta * sigma) / cost  =>  (beta * sigma - mu) / cost
    
    Alternatively, we can use a cost-weighted combination that favors
    high uncertainty (sigma) and low mean (mu), normalized by cost.
    
    Let's refine the LCB approach for maximization:
    We want to find points where the true value is likely low.
    LCB(x) = mu(x) - beta * sigma(x).
    We want to minimize LCB(x) subject to cost.
    So the "value" of a point is -LCB(x) / cost(x) = (beta * sigma(x) - mu(x)) / cost(x).
    
    Adaptive beta:
    Beta should be large when we have few observations (high uncertainty) and
    small when we have many (low uncertainty).
    However, we don't know the number of observations directly, only the current
    mu and sigma.
    A common heuristic is to scale beta with the maximum sigma observed in the
    current batch, or use a fixed beta that balances exploration/exploitation.
    Given the small budget (30 cost), exploration is crucial.
    A beta around 1.0 to 2.0 is typical for LCB.
    
    Let's try: score = (beta * sigma - mu) / cost
    With beta = 1.5 (aggressive exploration).
    
    Also, we must ensure finite outputs.
    """
    t0 = time.time()
    deadline = t0 + 290.0  # safety margin before 300s limit

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against zero or negative sigma and cost
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Adaptive exploration coefficient
    # We can make beta slightly adaptive based on the spread of sigma
    # If sigma is generally high, we might want a larger beta to explore more.
    # But a fixed beta is often robust. Let's use 1.5.
    beta = 1.5
    
    # LCB score: we want to maximize (beta * sigma - mu) / cost
    # This is equivalent to minimizing (mu - beta * sigma) * cost
    # But dividing by cost is the standard cost-aware adjustment.
    
    # Calculate the raw LCB-like score
    # Higher score means more promising.
    # If mu is low (good) and sigma is high (uncertain), score is high.
    # Wait: (beta * sigma - mu) / cost
    # If mu is very low (e.g., -3.8), -mu is large positive. Score is high.
    # If sigma is high, beta*sigma is large positive. Score is high.
    # This seems correct for a minimization problem framed as maximization of "potential improvement".
    
    score = (beta * sigma_safe - mu) / cost_safe

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    # Time guard: if we're close to deadline, return immediately
    if time.time() > deadline - 5.0:
        return score

    return score
# EVOLVE-BLOCK-END
