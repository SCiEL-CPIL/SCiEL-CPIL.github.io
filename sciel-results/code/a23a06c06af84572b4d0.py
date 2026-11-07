import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of the Beale function.
    Combines Expected Improvement (EI) with an adaptive UCB exploration term.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)

    # Standardized improvement
    z = (f_best - mu) / sigma

    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Adaptive beta for UCB
    # Beale function: min value 0, typical values 0-100+
    # Start with high exploration when f_best is large (poor), reduce as we improve
    # Use a smooth transition
    # f_best = 100 -> beta ~ 6.0
    # f_best = 10 -> beta ~ 3.0
    # f_best = 1 -> beta ~ 1.5
    # f_best = 0 -> beta ~ 1.0
    
    # Linear interpolation in log space for smoother behavior
    # beta = beta_min + (beta_max - beta_min) * exp(-f_best / scale)
    # At f_best=0: beta = beta_min + (beta_max - beta_min) = beta_max
    # At f_best=100: beta ≈ beta_min
    
    beta_min = 1.0
    beta_max = 7.0
    scale = 15.0
    
    beta = beta_min + (beta_max - beta_min) * np.exp(-f_best / scale)
    beta = float(np.clip(beta, beta_min, beta_max))

    # UCB term: we want to maximize, so we add beta * sigma
    # But EI already rewards low mu. The UCB term rewards high sigma.
    # Combined: EI + alpha * beta * sigma
    # Let's use a modest alpha to not over-explore
    alpha = 0.5
    score = ei + alpha * beta * sigma

    # Ensure all outputs are finite
    score = np.where(np.isfinite(score), score, -1e10)

    return score
