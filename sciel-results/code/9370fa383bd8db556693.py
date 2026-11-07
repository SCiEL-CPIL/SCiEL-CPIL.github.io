import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    and Lower Confidence Bound (LCB) for Bayesian Optimization.
    
    Parameters:
    mu (np.array): Mean of the Gaussian Process.
    sigma (np.array): Standard deviation of the Gaussian Process.
    f_best (float): Best observed function value so far.
    cost (np.array): Cost of evaluating each candidate point.
    
    Returns:
    np.array: Acquisition scores for each candidate point (higher is better).
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Ensure sigma is positive for division
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Expected Improvement (EI)
    # EI measures the expected improvement over f_best.
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Lower Confidence Bound (LCB) component
    # For minimization, LCB = mu - beta * sigma.
    # We want to maximize the "improvement" potential, so we use -LCB = beta * sigma - mu.
    # This term rewards low predicted means and high uncertainty.
    beta = 2.0
    lcb_score = beta * sigma - mu
    
    # 3. Hybrid Score
    # Combine EI and LCB scores.
    # EI drives exploitation near the current best.
    # LCB drives exploration in uncertain regions.
    score = ei + lcb_score
    
    # 4. Cost Awareness
    # Divide by cost to prefer cheaper evaluations.
    cost_safe = np.maximum(cost, 1e-6)
    score = score / cost_safe
    
    # 5. Sanitize
    # Ensure all scores are finite.
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
