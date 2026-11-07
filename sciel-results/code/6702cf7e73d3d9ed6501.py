"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = _start_time + 290.0  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Expected Improvement (EI)
    # EI = E[max(f_best - f(x), 0)] / cost
    # This is a standard Bayesian optimization acquisition function
    # that balances exploration (uncertainty) and exploitation (low predicted value)
    # while accounting for evaluation cost.
    
    # Compute z-score: how far below f_best the mean is, in units of sigma
    z = (f_best - mu) / (sigma + 1e-9)
    
    # Expected improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-adjusted score: higher EI per unit cost is better
    score = ei / (cost + 1e-9)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
