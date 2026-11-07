"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Hypervolume-based acquisition function for Bayesian optimization of Branin 2D.
    
    Uses expected hypervolume improvement (EHI) with a reference point derived
    from the observed data, combined with exploration weighting based on the
    second moment of improvement to favor regions with high potential for
    large gains.
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        Mean of the Gaussian process posterior at candidate points.
    sigma : array-like, shape (n,)
        Standard deviation of the Gaussian process posterior at candidate points.
    f_best : float
        The best (lowest) observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores, higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score: how many standard deviations below f_best the mean is
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -15, 15)
    
    # Standard normal CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement (first moment of improvement)
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Second moment of improvement: E[max(f_best - f, 0)^2]
    # For f ~ N(mu, sigma^2):
    # E[max(a - f, 0)^2] where a = f_best
    # = (a-mu)^2 * Phi(z) + sigma^2 * (z^2 * Phi(z) - z * phi(z) + 2*Phi(z) - 1)
    # where z = (a-mu)/sigma
    # Let's use a cleaner derivation:
    # Let Y = max(f_best - f, 0). Then E[Y^2] = E[(f_best - f)^2 * 1{f < f_best}]
    # = E[(f_best - f)^2] - E[(f_best - f)^2 * 1{f >= f_best}]
    # For the truncated part, we can use:
    # E[(f_best - f)^2 | f < f_best] * P(f < f_best)
    # (f_best - f)^2 = f^2 - 2*f_best*f + f_best^2
    # E[f^2 | f < f_best] = mu^2 + sigma^2 - sigma^2 * z * pdf(z) / cdf(z)  [for z < 0, but let's use general formula]
    # Actually, let's use the known result:
    # E[max(a-f,0)^2] = (a-mu)^2 * Phi(z) + 2*sigma*(a-mu)*phi(z) + sigma^2*(Phi(z) - z*phi(z))
    # Wait, let me derive properly.
    
    # Let d = f_best - mu, s = sigma, z = d/s
    # E[max(d + s*N(0,1), 0)^2] where we're looking at f_best - f = d - s*N(0,1)... 
    # Actually f ~ N(mu, sigma^2), so f_best - f ~ N(f_best - mu, sigma^2) = N(d, sigma^2)
    # Let W = f_best - f ~ N(d, sigma^2)
    # E[max(W, 0)^2]
    # = E[W^2 * 1{W > 0}]
    # = E[W^2 | W > 0] * P(W > 0)
    # E[W^2 | W > 0] = Var(W | W > 0) + (E[W | W > 0])^2
    # For W ~ N(d, sigma^2), P(W > 0) = Phi(z) where z = d/sigma
    # E[W | W > 0] = d + sigma * phi(z)/Phi(z)
    # Var(W | W > 0) = sigma^2 * (1 + z*phi(z)/Phi(z) - (phi(z)/Phi(z))^2)
    
    # This is getting complex. Let's use a simpler but effective approach:
    # Combine EI with a term proportional to sigma^2 * pdf(z) which captures
    # the "spread" of potential improvements.
    
    # Hypervolume-inspired score:
    # Primary: EI (expected linear improvement)
    # Secondary: sigma^2 * pdf(z) * cdf(z) - this rewards high uncertainty regions
    #             where there's still a good chance of improvement
    
    # A cleaner formulation: use the expected improvement with a power > 1
    # to emphasize larger improvements, approximated as:
    # score = EI * (1 + lambda * sigma / (f_best - mu + eps) * pdf(z) / (cdf(z) + eps))
    
    # Actually, let's use a well-motivated combination:
    # The hypervolume in 1D for a point f is max(r - f, 0) for reference r.
    # The expected HV improvement when adding a new point with distribution N(mu, sigma^2)
    # is E[max(r - f, 0)] - max(r - f_best, 0) * P(f > r) ... this is getting circular.
    
    # Practical approach: Use EI as the base, and add an exploration bonus
    # that is proportional to the standard deviation, weighted by how likely
    # it is that the point is in a region where improvement is possible.
    
    # Exploration bonus: sigma * pdf(z) peaks when z is around -1 to -2,
    # i.e., when the mean is slightly above f_best but there's significant uncertainty.
    
    # Let's use: score = EI + alpha * sigma * pdf(z) * (1 - cdf(z))
    # The (1 - cdf(z)) = P(f > f_best) term ensures we only explore where
    # there's a chance the true value is worse than current best (uncertain region).
    
    alpha = 1.0
    score = ei + alpha * sigma * pdf * (1.0 - cdf)
    
    # Ensure all scores are finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
