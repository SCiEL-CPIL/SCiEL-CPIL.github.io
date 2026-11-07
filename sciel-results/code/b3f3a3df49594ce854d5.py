import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of a fixed seeded random RBF mixture in 2D.
    
    Uses a combination of:
    1. Expected Improvement (EI) for exploitation
    2. A time-aware exploration bonus that decays over the course of the optimization
    
    The key insight is that we need to balance exploration (early) and exploitation (late).
    Since we don't know the step number directly, we use the relationship between 
    f_best and the expected range of the function to infer progress.
    
    For minimization:
    - EI = (f_best - mu) * Phi(Z) + sigma * phi(Z) where Z = (f_best - mu) / sigma
    - Higher EI means higher probability of improving upon f_best
    
    We add an exploration term that is larger when sigma is large relative to the 
    improvement, encouraging exploration in uncertain regions.
    """
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute Z-score for Expected Improvement
    Z = (f_best - mu) / sigma
    
    # Clip Z to avoid numerical issues
    Z = np.clip(Z, -10, 10)
    
    # Phi(Z) = standard normal CDF, phi(Z) = standard normal PDF
    # Using stable computation
    from scipy.stats import norm
    Phi = norm.cdf(Z)
    phi = norm.pdf(Z)
    
    # Expected Improvement for minimization
    EI = (f_best - mu) * Phi + sigma * phi
    
    # Exploration bonus: encourage exploring regions with high uncertainty
    # Scale by the relative uncertainty
    # The term (f_best - mu) / sigma is Z, so high |Z| means either very promising or very bad
    # We want to explore when uncertainty is high relative to potential improvement
    
    # A simple exploration term: proportional to sigma, but scaled down when we're 
    # already in a promising region (low mu relative to f_best)
    # Use a soft cap to prevent excessive exploration
    
    # Exploration weight that decays as we find better solutions
    # Estimate progress: if f_best is much lower than the typical scale, we're in exploitation phase
    # Typical scale of the function: we can estimate from the spread of mu
    mu_scale = np.std(mu) + 1e-8
    
    # Progress estimate: how much lower is f_best compared to the mean of mu
    # If f_best is close to min(mu), we're in exploitation
    progress = np.clip((np.mean(mu) - f_best) / (2 * mu_scale), 0, 1)
    
    # Exploration coefficient: starts high, decreases as we progress
    beta_explore = 0.5 * (1 - progress)
    
    # Combine EI with exploration
    # The exploration term should be larger when sigma is large
    exploration_bonus = beta_explore * sigma
    
    scores = EI + exploration_bonus
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure scores are reasonably bounded
    scores = np.clip(scores, -1e6, 1e6)
    
    return scores
