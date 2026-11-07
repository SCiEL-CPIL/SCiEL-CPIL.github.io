# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 295.0  # safety margin before 300s hard limit


def acquisition(mu, sigma, f_best):
    """
    Quantile-based acquisition function for minimization.
    
    Uses the 0.1st percentile (Q0.1) of the posterior predictive distribution
    as the primary score. This targets the lower tail of the uncertainty
    distribution, optimizing the probability of finding a value significantly
    below the current best.
    
    Q0.1 = mu - z_0.9 * sigma, where z_0.9 = Phi^{-1}(0.9) ≈ 1.2816
    
    A small Expected Improvement (EI) component is added as a refinement
    to provide gradient information when the quantile signal is flat,
    helping to distinguish among candidates with similar lower-tail values.
    
    The combination uses a soft-max-like blending: primarily Q0.1 with a
    small EI boost, ensuring the lower-tail targeting remains dominant.
    """
    # Time guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Return a neutral score if time is critical
        return np.zeros_like(mu)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Guard against degenerate / negative std
    s = np.clip(sigma, 1e-9, None)
    
    # Q0.1 = mu - 1.2816 * sigma (lower tail target)
    # Maximizing Q0.1 means we want the highest lower-tail value,
    # i.e., we want points where even the 0.1st percentile is high.
    # Wait - for minimization, we want LOW values. So we want to MINIMIZE
    # the quantile. But the acquisition should return HIGHER score for 
    # better candidates. So we negate: score = -Q0.1 = -mu + 1.2816*sigma
    # Actually, let me reconsider: we want to find points with low f values.
    # The 0.1st percentile of the posterior at a point x is:
    #   Q0.1(x) = mu(x) - 1.2816 * sigma(x)
    # A low Q0.1 means there's a good chance of finding a very low value.
    # So we want to MINIMIZE Q0.1, which means we want to MAXIMIZE -Q0.1.
    # score = -Q0.1 = -mu + 1.2816 * sigma
    
    z_01 = 1.2815515655446004  # Phi^{-1}(0.9), so Q0.1 = mu - z_01 * sigma
    
    # Primary score: negative of the 0.1st percentile (higher is better for minimization)
    q01_score = -mu + z_01 * s
    
    # Compute EI as a refinement signal
    z = (f_best - mu) / s
    z = np.clip(z, -38.0, 38.0)
    
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf + s * pdf
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Normalize EI to be on a comparable scale to q01_score
    # Use a small weight for EI to keep Q0.1 dominant
    ei_max = np.max(ei) if np.max(ei) > 1e-12 else 1.0
    q01_range = np.max(q01_score) - np.min(q01_score) if np.max(q01_score) > np.min(q01_score) + 1e-12 else 1.0
    
    # Scale EI to contribute at most ~10% of the Q0.1 range
    ei_weighted = 0.1 * (ei / ei_max) * q01_range
    
    # Combine: primarily Q0.1-based score with small EI refinement
    scores = q01_score + ei_weighted
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
