"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    GP-UCB with time-varying beta_t = sqrt(2t log(1+t)) combined with normalized EI.
    
    mu, sigma: arrays of shape (2000,) with posterior mean and std in objective units.
    f_best: lowest observed value so far.
    
    Since acquisition is stateless, we estimate the current evaluation count t from
    the spread of sigma. Early in optimization, sigma values are large (few observations),
    while later they become smaller. We use the median sigma as a proxy for t.
    
    For the Beale function on [-4.5, 4.5]^2, the initial sigma (before any evaluations)
    would be related to the function scale. With 5 initial points, sigma is still relatively
    high. We map observed sigma statistics to an estimated t in [5, 30].
    
    We estimate t by looking at the fraction of points where sigma is below a threshold.
    Alternatively, we can use a simpler heuristic: the mean sigma relative to the range
    of mu gives a sense of how much we've learned.
    
    A robust approach: use the 90th percentile of sigma. Early on, most points have high
    uncertainty; as we collect more data, uncertainty decreases.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Estimate t from sigma distribution
    # When t is small (few evals), sigma is large across most of the domain
    # When t is large, sigma is small except near unexplored regions
    sigma_90 = np.percentile(sigma_safe, 90)
    sigma_10 = np.percentile(sigma_safe, 10)
    
    # The ratio sigma_90/sigma_10 indicates how much exploration remains
    # If all sigma are similar, we're still in early stages
    # Use a simple mapping: estimate t based on the overall scale of sigma
    # For Beale function, f ranges from 0 to ~180, so initial sigma ~ O(10-50)
    # After many evals, sigma drops significantly
    
    # Heuristic: map median sigma to t
    # sigma ~ 20 -> t ~ 5, sigma ~ 1 -> t ~ 30 (rough calibration)
    med_sigma = np.median(sigma_safe)
    # Log-scale mapping
    t_est = 5 + 25 * (1.0 - np.clip(np.log1p(med_sigma) / np.log1p(20.0), 0.0, 1.0))
    t_est = np.clip(t_est, 5.0, 30.0)
    
    # Compute beta_t = sqrt(2 * t * log(1 + t))
    beta = np.sqrt(2.0 * t_est * np.log1p(t_est))
    
    # UCB term: mu + beta * sigma
    ucb = mu + beta * sigma_safe
    
    # Expected Improvement term
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize EI to be on similar scale
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # Combine UCB and normalized EI
    # Weight EI to provide exploitation pressure without dominating
    score = ucb + 0.3 * ei_norm
    
    return score
# EVOLVE-BLOCK-END
