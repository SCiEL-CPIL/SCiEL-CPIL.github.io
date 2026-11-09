"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Thompson Sampling with increased exploration factor.
    # The Beale function has a narrow global minimum. The GP length scale (1.8 in original units)
    # may lead to overconfidence in unexplored regions, causing premature convergence.
    # We amplify the standard deviation to encourage more exploration.
    
    z = np.random.standard_normal(len(mu))
    
    # Exploration factor to balance exploitation and exploration
    kappa = 1.5
    samples = mu + kappa * sigma * z
    
    # Higher score means evaluate next; for minimization, lower sample is better
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
