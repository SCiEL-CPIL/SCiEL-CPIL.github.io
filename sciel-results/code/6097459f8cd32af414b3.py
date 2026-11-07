"""Direct cost-aware acquisition for Bayesian optimization on griewank 2d."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function: Cost-Adjusted Lower Confidence Bound (LCB).
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions in objective units
    - sigma: array of shape (2000,) - GP std-dev predictions in objective units
    - f_best: float - best observed value so far
    - cost: array of shape (2000,) - evaluation costs (1 + mean of scaled coords)
    
    Returns:
    - scores: array of shape (2000,) of finite scores, higher = more promising
    """
    # Deadline guard: track elapsed time
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Dynamically determine kappa based on budget consumption
    # cost array contains individual evaluation costs; we need total cost spent.
    # However, the signature only provides individual costs for candidates.
    # The prompt says "While total cost is below 30...".
    # We don't have the exact total cost spent so far in the function signature directly,
    # but we can infer the phase from f_best or just use a fixed schedule.
    # Let's look at the "cost" parameter. It is shape (2000,). These are costs of *candidates*.
    # We don't know the history.
    # Standard approach: Use a fixed kappa or estimate progress.
    # Since we don't have the total spent cost, we can't dynamically adjust kappa based on *spent* budget easily
    # unless we assume a typical trajectory or use a heuristic.
    
    # Alternative: The directive suggests "kappa is dynamically increased as the total budget is consumed".
    # Without the total spent cost, we can approximate the phase by the number of evaluations if we had it,
    # but we don't.
    # Let's stick to a robust fixed kappa or a simple heuristic.
    # A common choice for LCB in BO is kappa ~ 2.0.
    # To incorporate cost, we divide by cost.
    
    # Let's try a fixed kappa first, as dynamic adjustment without history is risky.
    # However, the prompt mentions "total cost is below 30".
    # Maybe we can estimate the phase? No, we can't.
    
    # Let's use a fixed kappa = 2.0.
    kappa = 2.0
    
    # LCB = mu - kappa * sigma (lower bound on the function value)
    # We want to minimize the function, so lower LCB is better.
    # But we want to maximize the acquisition score.
    # So score = -LCB / cost = (kappa * sigma - mu) / cost
    # Or simply: score = (mu - kappa * sigma) / cost, but we want higher scores for better candidates.
    # If we minimize f, we want low f.
    # LCB gives a lower bound. A very low LCB means high potential for improvement.
    # So we want to maximize (kappa * sigma - mu) / cost?
    # Let's check signs.
    # If mu is low (good), -mu is high.
    # If sigma is high (uncertain), kappa*sigma is high.
    # So (kappa * sigma - mu) is high for promising points.
    # Dividing by cost penalizes expensive points.
    
    lcb_score = (kappa * sigma_safe - mu) / np.maximum(cost, 1e-9)
    
    # Ensure all scores are finite
    scores = np.nan_to_num(lcb_score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline guard: if we're close to timeout, return immediately
    elapsed = time.time() - start_time
    if elapsed > 280:  # Safety margin before 300s limit
        return scores
    
    return scores
# EVOLVE-BLOCK-END
