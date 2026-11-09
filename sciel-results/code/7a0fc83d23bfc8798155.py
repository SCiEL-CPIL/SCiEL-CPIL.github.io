"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # 300s limit, keep 10s margin

def acquisition(mu, sigma, f_best):
    # Check time budget
    if time.time() > _DEADLINE:
        return np.zeros_like(mu)
    
    # Expected improvement with exploration parameter xi
    xi = 0.05
    improvement = f_best - xi - mu
    z = improvement / (sigma + 1e-9)
    return improvement * norm.cdf(z) + sigma * norm.pdf(z)
# EVOLVE-BLOCK-END
