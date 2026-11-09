"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    # Thompson Sampling: draw one sample from the posterior predictive at each candidate.
    # The GP posterior at each point is Normal(mu_i, sigma_i^2).
    # Since we are minimizing, we want the lowest sampled value.
    # Score = negative of the sample, so higher score means lower sampled value.
    #
    # To avoid re-drawing 2000 independent z's each call (which would make the
    # acquisition non-deterministic in a way that could hurt), we draw a single
    # global random draw and use it consistently. However, the standard Thompson
    # Sampling for BO draws one sample per candidate point independently.
    #
    # We use a fixed seed per call? No — we need fresh randomness each call to
    # avoid always picking the same point. We draw fresh standard normals.
    
    # Draw standard normal samples for each candidate
    z = np.random.standard_normal(mu.shape)
    
    # Sample from the posterior predictive: y_i ~ N(mu_i, sigma_i^2)
    # For minimization, lower is better, so we want to maximize -y_i
    sample = mu - sigma * z
    
    # Return negative of the sample as the score (higher = better = lower sampled value)
    scores = -sample
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e18)
    
    return scores
# EVOLVE-BLOCK-END
