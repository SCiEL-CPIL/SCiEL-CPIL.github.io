"""Cost-aware LCB acquisition for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Adjusted Lower Confidence Bound (LCB) acquisition function.
    
    score = mu + kappa * sigma - lambda_cost * cost
    
    This directly targets the minimum observed value while accounting for
    evaluation cost. Higher scores indicate more desirable points to evaluate.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions at 2000 candidate points.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions at 2000 candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable).
    """
    # Track elapsed time to stay within budget
    t0 = time.time()
    deadline = 290.0  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Adaptive kappa based on progress
    # Early on: more exploration (higher kappa)
    # Later: more exploitation (lower kappa)
    # Use f_best as a proxy for progress; as f_best approaches 0, reduce kappa
    
    # Normalize f_best to [0, 1] range for adaptive behavior
    # f_best starts around 1-2 and approaches 0
    progress = np.clip(1.0 - f_best / 2.0, 0.0, 1.0) if f_best > 0 else 1.0
    
    # kappa: exploration parameter
    # Start with high exploration, decrease as we converge
    kappa = 2.5 * (1.0 - 0.5 * progress)
    
    # lambda_cost: cost penalty parameter
    # Higher cost should be penalized more
    lambda_cost = 0.5
    
    # Compute LCB score: mu + kappa * sigma - lambda_cost * cost
    # Since we want to MINIMIZE f, lower mu is better, so we use -mu + kappa*sigma
    # Wait: LCB for minimization is mu - kappa * sigma (lower bound on true value)
    # But we want HIGHER score = better. So score = -mu + kappa * sigma - lambda_cost * cost
    # Actually, let's think again:
    # For minimization, we want points where the lower confidence bound is low.
    # LCB = mu - kappa * sigma (this is a lower bound on the true f(x))
    # We want to maximize the probability of finding a new minimum.
    # So we want low LCB values, which means we want to MINIMIZE LCB.
    # But our score should be HIGHER for better points.
    # So score = -LCB + bonus = -(mu - kappa*sigma) + bonus = -mu + kappa*sigma + bonus
    # And we subtract cost penalty.
    
    score = -mu + kappa * sigma - lambda_cost * cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Check time
    if time.time() - t0 > deadline:
        return score
    
    return score
# EVOLVE-BLOCK-END
