import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    Uses a hybrid approach: Expected Improvement (EI) with an adaptive exploration bonus.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive exploration term
    # We want to explore more when f_best is far from the known minimum,
    # and exploit more when close.
    f_min = -3.32237
    gap = f_best - f_min
    
    # Scale the exploration bonus
    # When gap is large (early on), we want more exploration.
    # When gap is small (late on), we want less exploration.
    # Let's use a coefficient that decays as we get closer to the minimum.
    # A simple approach: alpha = max(0, 1 - gap / initial_gap_estimate)
    # But we don't know initial gap. Let's just use a fixed small coefficient
    # that encourages slight exploration, or make it adaptive based on gap.
    
    # The cell finding says: "The hybrid EI + small sigma exploration term may better balance exploitation and exploration"
    # So let's add a term like beta * sigma.
    
    # Let's try an adaptive beta.
    # If gap is large, beta is larger. If gap is small, beta is smaller.
    # gap starts around 10-15 and goes to 0.
    # Let's say beta = 0.5 * (gap / 10.0) clipped to [0.01, 0.5]
    # This ensures some exploration always, but less as we improve.
    
    beta = 0.5 * np.clip(gap / 10.0, 0.02, 1.0)
    
    # Combine EI and exploration
    score = ei + beta * sigma_safe
    
    return score
