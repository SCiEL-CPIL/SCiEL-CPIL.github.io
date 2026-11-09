"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Thompson Sampling Strategy
    # Instead of a deterministic optimistic score (UCB) or probability-weighted improvement (EI),
    # we draw a sample from the posterior distribution of the objective function at each candidate.
    # We then select the candidate with the lowest sampled value (for minimization).
    # Since the acquisition function must return higher scores for better candidates,
    # we return the negative of the sampled value.
    
    # Ensure sigma is non-negative and finite to prevent numerical errors in sampling
    sigma = np.maximum(sigma, 1e-9)
    
    # Generate standard normal random variables for the 2000 candidates
    # Using a fixed seed is not possible as we don't have state, but random sampling is the core of TS.
    # The randomness helps escape local optima in the Ackley function.
    z = np.random.normal(0, 1, size=mu.shape)
    
    # Sample from the posterior: y ~ N(mu, sigma^2)
    y = mu + sigma * z
    
    # For minimization, lower y is better.
    # Acquisition score should be higher for better candidates.
    # Therefore, score = -y
    score = -y
    
    # Ensure finite outputs
    score = np.nan_to_num(score, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
