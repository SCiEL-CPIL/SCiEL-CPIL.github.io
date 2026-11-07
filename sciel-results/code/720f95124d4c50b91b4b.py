"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) with beta=2.0
    # Balances exploitation (mu) and exploration (sigma)
    beta = 2.0
    return mu + beta * sigma
# EVOLVE-BLOCK-END
