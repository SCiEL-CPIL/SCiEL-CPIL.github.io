"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Use Probability of Improvement (POI) per unit cost.
    # POI is more aggressive in exploration than EI when sigma is high,
    # which can help escape local optima in the Rosenbrock function.
    # We also add a small exploration bonus based on sigma to encourage
    # visiting uncertain regions, but weighted by cost efficiency.
    
    # Probability of improvement: P(f(x) < f_best)
    z = (f_best - mu) / (sigma + 1e-9)
    poi = norm.cdf(z)
    
    # Add a small exploration term to balance exploitation and exploration
    # The exploration term is proportional to sigma, but we keep it small
    # to not completely dominate the exploitation signal.
    exploration = 0.1 * sigma / (sigma + 1e-9)
    
    # Combine POI and exploration
    score = poi + exploration
    
    # Divide by cost to get cost-aware acquisition
    score = score / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
