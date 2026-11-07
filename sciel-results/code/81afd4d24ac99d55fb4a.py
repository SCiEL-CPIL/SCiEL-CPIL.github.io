"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization
    # We maximize -mu + beta * sigma
    # Reduced exploration weight (beta=0.2) to favor exploitation
    # as suggested by the insight that reducing exploration helped.
    beta = 0.2
    return -mu + beta * sigma
# EVOLVE-BLOCK-END
