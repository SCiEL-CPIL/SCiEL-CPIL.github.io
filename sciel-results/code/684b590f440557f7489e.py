import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-adjusted Lower Confidence Bound (LCB)
    # LCB(x) = mu(x) - beta(x) * sigma(x)
    # We want to minimize LCB, so we maximize -LCB(x) = -mu(x) + beta(x) * sigma(x)
    # To make it cost-aware, we scale the exploration term (sigma) by 1/cost.
    # This penalizes high-cost regions in the uncertainty bonus.
    # Score = -mu + (sigma / cost)
    
    # Ensure cost is not zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # The beta factor: 
    # In standard UCB/LCB for BO, beta is often a constant or depends on the number of evaluations.
    # Since we don't have the step count directly, we can use a heuristic.
    # However, the prompt suggests "exploration term is scaled by the inverse cost".
    # Let's try a simple formulation first: Score = -mu + sigma / cost
    
    # But wait, mu and sigma are in objective units. Cost is dimensionless (1 to 2).
    # If f is large, sigma might be small relative to mu?
    # Let's look at the scales.
    # Rosenbrock min is 0. f_best starts high.
    # Initially, mu ~ 0, sigma ~ 1 (standardized targets? No, "standardized targets" means y' = (y - mean) / std).
    # Wait, "mu and sigma ... are in objective units". This implies the GP output is NOT standardized in the returned mu/sigma, or it is un-standardized back.
    # "GP: ... standardized targets; mu and sigma ... are in objective units."
    # This means the caller passes mu and sigma already scaled back to the original f(x) scale.
    
    # So mu and sigma have the same units as f(x).
    # Cost is in [1, 2].
    
    # If we use Score = -mu + sigma / cost:
    # Where sigma is large (uncertain), score is high (good for exploration).
    # Where cost is high, the sigma term is reduced.
    # Where mu is low, score is high (good for exploitation).
    
    # Is the scale of mu and sigma comparable?
    # Early on, f_best is large (e.g., 100). mu is around 100. sigma is large.
    # Later, f_best is small (e.g., 0.1). mu is around 0.1. sigma is small.
    # The term sigma/cost will be small later.
    # The term -mu will dominate later. This is good for convergence.
    
    # Early on, -mu is large negative (since mu is positive and large for most points, except near the minimum which is shifted).
    # Actually, for Rosenbrock, most points have high f.
    # So -mu is a strong signal to go to low mu regions.
    # sigma/cost adds exploration.
    
    # Let's refine the beta.
    # A constant beta=1 might be too weak or too strong.
    # In UCB, beta is often ~ sqrt(2 * log(...)).
    # But since we are returning a score, and the evaluator picks the max, the absolute scale doesn't matter for ranking, only the relative trade-off between mu and sigma.
    
    # Let's try: Score = -mu + sigma / cost
    
    scores = -mu + sigma / cost_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
