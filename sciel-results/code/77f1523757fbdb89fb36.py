"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Expected improvement with softened cost penalty.
    # The global minimum of the 3-hump camel function is in a higher cost region
    # than the local minimum, so a strong cost penalty can bias the search
    # towards the suboptimal local minimum. We use a weaker penalty.
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # cost is in [1, 2]. We use a linear penalty with slope 0.2.
    # This reduces the relative penalty for expensive regions compared to 1/cost.
    cost_penalty = 1.0 + 0.2 * (cost - 1.0)
    return ei / cost_penalty
# EVOLVE-BLOCK-END
