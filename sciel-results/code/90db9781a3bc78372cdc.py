"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Thompson Sampling with reduced exploration (scaled sigma)
    # The default TS (sigma * eps) implies a high exploration factor (~4 sigma for 2000 candidates).
    # For simple functions like sphere, less exploration might lead to faster convergence to the minimum.
    # We scale the uncertainty by a factor of 0.5 to favor exploitation more.
    
    # Standard Thompson Sampling: Sample from the posterior N(mu, sigma^2)
    # This balances exploration and exploitation naturally.
    samples = mu + sigma * np.random.standard_normal(mu.shape)
    
    # We want to minimize, so lower sampled value is better.
    # Return negative of the sample so that higher score = lower sampled objective.
    return -samples
# EVOLVE-BLOCK-END
