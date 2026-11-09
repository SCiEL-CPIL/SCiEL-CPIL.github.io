"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Use Upper Confidence Bound (UCB) for minimization.
    # This balances exploitation (low mu) and exploration (high sigma).
    # For multimodal functions like Weierstrass, this often outperforms EI.
    kappa = 2.5
    scores = -mu + kappa * sigma
    return scores
# EVOLVE-BLOCK-END
