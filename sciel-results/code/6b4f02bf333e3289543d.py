"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) acquisition function.
    
    KG estimates the expected reduction in the global minimum value
    by considering the joint distribution of the 2000 candidates.
    We use a Monte Carlo approximation with a small number of samples
    to estimate the KG efficiently.
    
    For each candidate x_i, KG(x_i) = E[min_j y_j - min_j y_j'] 
    where y_j is the posterior at candidate j, and y_j' is the posterior
    at candidate j after observing y_i at x_i.
    
    We approximate this with a small number of Monte Carlo samples.
    """
    n = len(mu)
    
    # Guard against zero or near-zero sigma
    sigma = np.maximum(sigma, 1e-9)
    
    # Number of MC samples for KG approximation
    # Keep this small for speed - we have a time budget
    n_samples = 10
    
    # Precompute some useful quantities
    # The current best value among the 2000 candidates
    # We want to estimate how much observing each candidate would improve our best
    
    # For efficiency, we'll use a vectorized MC approach
    # Sample from the joint posterior
    # y_i ~ N(mu_i, sigma_i^2)
    
    # For each candidate i, the KG is:
    # E_yi [ min_j (y_j | y_i) - min_j y_j ]
    
    # Since we can't easily compute the conditional min, we use the approximation:
    # KG_i ≈ E[ max(0, f_best - y_i) ] weighted by the probability that y_i is the new min
    # But a better approximation: consider that after observing y_i, the best
    # among the remaining candidates is min(min_{j≠i} y_j, y_i)
    
    # Simplified KG: For each candidate i, sample y_i from N(mu_i, sigma_i^2)
    # Then the new best is min(y_i, min_{j≠i} mu_j adjusted for uncertainty)
    
    # Even simpler and faster: use the "expected improvement of the minimum"
    # which is related to EI but considers the global min
    
    # Fast approximation: 
    # 1. Sample n_samples values from each candidate's posterior
    # 2. For each sample s, compute the min over all candidates
    # 3. For each candidate i, compute the expected improvement if we observe y_i
    
    # Actually, let's use a more direct and fast approach:
    # The KG for candidate i is approximately:
    # E[max(0, min_{j≠i} y_j - y_i)] where y_i ~ N(mu_i, sigma_i^2)
    # and min_{j≠i} y_j is approximated by its mean and std
    
    # For computational efficiency, let's use a hybrid:
    # - Compute EI for each candidate (fast, vectorized)
    # - Add a KG-like correction that rewards candidates that could be the global min
    
    # Step 1: Compute EI
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Step 2: Estimate the probability that each candidate is the global min
    # P(y_i < min_{j≠i} y_j)
    # Approximate: for each i, P(y_i < y_j) for all j≠i
    # This is expensive (O(n^2)), so we use a sampling-based approximation
    
    # Use a smaller set of reference points to estimate the min distribution
    # Sample from the posterior of all candidates
    rng = np.random.RandomState(42)
    
    # For speed, use a coarser grid for the min estimation
    # Sample n_samples from each candidate
    samples = mu[:, np.newaxis] + sigma[:, np.newaxis] * rng.randn(n, n_samples)
    
    # For each sample, find the min index
    min_indices = np.argmin(samples, axis=0)
    
    # Count how often each candidate is the min
    min_counts = np.bincount(min_indices, minlength=n).astype(float)
    p_min = min_counts / n_samples
    
    # The KG-like score: combine EI with the probability of being the global min
    # Candidates that are likely to be the global min get a boost
    # Also, candidates with high EI get a boost
    
    # Normalize EI to [0, 1] range for combining
    ei_max = np.max(ei)
    if ei_max > 0:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    # Combine: weight EI by (1 + p_min) to reward global-min candidates
    # This encourages exploration of regions that might contain the true global optimum
    score = ei * (1.0 + 2.0 * p_min)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
