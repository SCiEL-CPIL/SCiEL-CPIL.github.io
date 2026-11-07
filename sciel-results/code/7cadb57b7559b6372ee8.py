import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using CEI with tanh-scaled exploration in objective units.
    Structural change: no min-max normalization; works directly in objective units.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clip sigma to avoid division issues
    sigma = np.clip(sigma, 1e-8, None)
    
    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0)
    
    # Cost-Effective EI: improvement per unit cost
    cei = ei / cost
    
    # Exploration: uncertainty scaled by tanh of improvement potential
    # This gives high exploration weight to uncertain regions where
    # the GP mean suggests significant improvement over f_best
    # tanh saturates so that very large improvements don't dominate
    improvement = np.maximum(f_best - mu, 0)
    # Scale factor: when f_best is large (far from optimum), explore more
    # When f_best is small (close to optimum), explore less
    scale = 1.0 / (1.0 + f_best)
    explore = sigma * np.tanh(improvement / (sigma + 1e-8)) * scale / cost
    
    # Combine: CEI primary, exploration secondary
    scores = cei + 0.5 * explore
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
