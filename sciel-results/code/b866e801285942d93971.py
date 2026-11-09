"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Thompson Sampling: draw a single sample from the posterior predictive
    # at all candidates. Since we are minimizing, lower samples are better,
    # but the acquisition function must return scores where HIGHER means
    # evaluate next. So we negate the sample to convert minimization to maximization.
    # 
    # Posterior predictive: y ~ N(mu, sigma^2)
    # For minimization, we want low y, so we maximize -y.
    # Sample: y_sample = mu - sigma * z where z ~ N(0,1) (or mu + sigma * z)
    # Actually, for minimization with Thompson Sampling, we sample from the posterior
    # and pick the point with the lowest predicted value. Since the interface requires
    # higher scores = better, we return -sample.
    
    # Use a single random draw for all candidates
    z = np.random.standard_normal(len(mu))
    samples = mu + sigma * z
    
    # Higher score means evaluate next; for minimization, lower sample is better
    # So we negate the sample
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
