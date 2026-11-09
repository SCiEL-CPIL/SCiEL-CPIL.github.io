"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining LCB with budget-aware exploration scaling.
    """
    t0 = time.time()
    deadline = t0 + 250.0  # Safety margin before 300s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Lower Confidence Bound (LCB) for minimization
    # LCB = mu - beta * sigma (lower is better)
    # We want higher score = better, so we use -LCB = beta * sigma - mu
    # Budget-aware beta: scale exploration by remaining budget
    
    # Estimate remaining budget: total budget is 30, current cost is sum of cost array
    # But we don't know the exact current total cost from the inputs directly in a simple way
    # The 'cost' input is the cost of each candidate point.
    # We can use the mean cost or a fixed heuristic for the budget scaling.
    # Let's assume the total budget is 30.
    # The current total cost is not directly given, but we can approximate the "phase" of the search.
    # Early search: low total cost -> high beta. Late search: high total cost -> low beta.
    # Since we don't have the exact total cost, we can use the magnitude of f_best or just a fixed adaptive scheme.
    
    # Alternative: Use the variance of sigma to determine exploration need.
    # Or, simpler: use a fixed beta that is slightly lower than 2.0 to balance with EI.
    # The current program uses beta=2.0 and sqrt(cost) penalty.
    # Let's refine the beta to be adaptive based on the uncertainty relative to the improvement.
    
    # Calculate a dynamic beta based on the z-score
    # High |z| means we are confident about the improvement or lack thereof.
    # Low |z| means high uncertainty.
    # Let's keep beta relatively constant but tune it.
    beta = 1.8
    
    lcb_score = -mu + beta * sigma_safe

    # Blend EI and LCB
    # Use a smooth blend based on the z-score
    # When z is large positive (mu << f_best), EI is high, we should exploit.
    # When z is small or negative, LCB helps explore.
    # Let's use a fixed blend but optimized.
    w_ei = 0.6
    w_lcb = 0.4
    
    raw_score = w_ei * ei + w_lcb * lcb_score

    # Cost-aware: use sqrt(cost) penalty as in the current best program
    scores = raw_score / np.sqrt(cost_safe)

    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Check deadline
    if time.time() > deadline:
        return scores

    return scores
# EVOLVE-BLOCK-END
