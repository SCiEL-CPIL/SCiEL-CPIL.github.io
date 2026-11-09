"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Probability of improvement with adaptive uncertainty bonus
    # POI: P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # Add uncertainty term to encourage exploration of high-sigma regions
    # Scale uncertainty by current f_best to adapt to problem scale
    z = (f_best - mu) / (sigma + 1e-9)
    poi = norm.cdf(z)
    
    # Adaptive uncertainty bonus: larger when sigma is high relative to f_best
    # This encourages exploration in uncertain regions
    uncertainty_bonus = sigma / (abs(f_best) + 1e-6)
    
    # Combine POI and uncertainty, normalized by cost
    score = (poi + 0.5 * uncertainty_bonus) / cost
    
    # Ensure finite values
    return np.where(np.isfinite(score), score, 0.0)
# EVOLVE-BLOCK-END
