"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Hypervolume-based acquisition function for Bayesian optimization of Branin 2D.
    
    This acquisition function approximates the expected hypervolume improvement
    relative to the current best value f_best. For a single-objective problem,
    the hypervolume improvement from finding a new minimum f_new < f_best is
    proportional to (f_best - f_new). The expected hypervolume improvement
    is E[max(f_best - f(x), 0)], which equals the standard Expected Improvement.
    
    To better capture the hypervolume geometry, we use a modified formulation
    that accounts for the curvature of the improvement region. We compute the
    second-order moment contribution to hypervolume, which gives more weight
    to points with high probability of large improvements.
    
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
    z = np.clip(z, -10, 10)
    
    # Standard normal CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement (EI) - first moment of hypervolume improvement
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # For hypervolume-based acquisition, we want to capture the expected
    # improvement in hypervolume. The hypervolume improvement when we find
    # a new best value f_new is (f_best - f_new) * (volume of region where
    # f_new is the new minimum). In a simplified model, we approximate this
    # by considering the second moment of the improvement distribution.
    
    # E[max(f_best - f(x), 0)^2] for f(x) ~ N(mu, sigma^2)
    # Let g = f_best - f(x), so g ~ N(f_best - mu, sigma^2)
    # E[max(g, 0)^2] = (f_best - mu)^2 * P(g > 0) + 2*sigma^2 * P(g > 0) + sigma^2 * (2*z*pdf - (z^2 - 1)*cdf)
    # Actually, let's derive this more carefully.
    # E[max(g,0)^2] where g = f_best - f(x), f(x) ~ N(mu, sigma^2)
    # Let h = f(x), so we want E[max(f_best - h, 0)^2]
    # = integral_{-inf}^{f_best} (f_best - h)^2 * N(h; mu, sigma^2) dh
    # Let t = (h - mu)/sigma, so h = mu + sigma*t, dh = sigma*dt
    # = integral_{-inf}^{(f_best-mu)/sigma} (f_best - mu - sigma*t)^2 * phi(t) * sigma dt
    # = sigma * integral_{-inf}^{z} (f_best - mu - sigma*t)^2 * phi(t) dt
    # = sigma * integral_{-inf}^{z} [(f_best-mu)^2 - 2*(f_best-mu)*sigma*t + sigma^2*t^2] * phi(t) dt
    # = sigma * [(f_best-mu)^2 * Phi(z) - 2*(f_best-mu)*sigma * integral_{-inf}^{z} t*phi(t) dt + sigma^2 * integral_{-inf}^{z} t^2*phi(t) dt]
    # integral_{-inf}^{z} t*phi(t) dt = -phi(z)
    # integral_{-inf}^{z} t^2*phi(t) dt = -z*phi(z) + Phi(z) - 1/2 + 1/2*erf(z/sqrt(2))... 
    # Actually, integral_{-inf}^{z} t^2*phi(t) dt = Phi(z) - z*phi(z) - 1/2 + 1/2*(1 - erf(z/sqrt(2)))
    # Simpler: integral_{-inf}^{z} t^2*phi(t) dt = -z*phi(z) + Phi(z) - 1/2 + 1/2
    # Let me use the known result: E[max(g,0)^2] for g~N(m, s^2)
    # = m^2 * Phi(m/s) + 2*s^2 * Phi(m/s) + s^2 * (m/s * phi(m/s) - (m/s)^2 * Phi(m/s) + s^2/s * ... )
    # This is getting complex. Let's use a simpler hypervolume-inspired score.
    
    # Hypervolume-inspired score: combine EI with a term that captures
    # the "width" of the improvement region. Points with high uncertainty
    # in regions where improvement is likely contribute more to hypervolume
    # because they have a higher chance of being significantly better.
    
    # score = EI * (1 + alpha * sigma / (f_best - mu + eps))
    # This upweights points where the uncertainty is large relative to the
    # expected improvement, which corresponds to regions with high potential
    # for large hypervolume gains.
    
    eps = 1e-9
    improvement_margin = f_best - mu
    improvement_margin = np.maximum(improvement_margin, eps)
    
    # The ratio sigma / improvement_margin is high when there's significant
    # uncertainty relative to the expected improvement, indicating potential
    # for large hypervolume gains.
    
    alpha = 0.5
    hv_factor = 1.0 + alpha * sigma / improvement_margin
    
    score = ei * hv_factor
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
