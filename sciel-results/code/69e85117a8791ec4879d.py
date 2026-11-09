"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Precompute constants for speed
_EPS = 1e-12
_DEADLINE_MARGIN = 10.0  # seconds to spare before the 330s limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a time-aware adaptive combination of EI and UCB.
    
    Strategy:
    1. Compute Expected Improvement (EI) for exploitation.
    2. Compute UCB with a dynamic kappa for exploration.
    3. Blend them based on the remaining uncertainty relative to the best observed value.
    
    This structural change moves from pure Thompson Sampling (stochastic) to a 
    deterministic hybrid that adapts its exploration-exploitation trade-off 
    dynamically based on the posterior uncertainty landscape.
    
    A time guard ensures we return before the deadline.
    """
    start_time = time.time()
    
    n = len(mu)
    
    # Ensure inputs are valid
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        # Fallback to a simple UCB if inputs are corrupted
        return -mu + 2.0 * sigma
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, _EPS)
    
    # Compute EI (Expected Improvement) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is the CDF and phi is the PDF of the standard normal distribution
    
    # Handle the case where f_best is very large or mu/sigma are extreme
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in the normal CDF/PDF
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # EI is always non-negative; higher is better
    
    # Compute UCB with a dynamic kappa
    # The idea: use more exploration when uncertainty is high relative to the gap to f_best
    # kappa = beta * (sigma / (f_best - min(mu) + eps))
    # But let's use a simpler adaptive scheme:
    # kappa = 2.0 * (median(sigma) / (f_best - min(mu) + eps))
    
    mu_min = np.min(mu)
    sigma_median = np.median(sigma)
    gap = f_best - mu_min
    if gap > 1e-10:
        kappa = 2.0 * (sigma_median / gap)
    else:
        kappa = 2.0
    
    # Cap kappa to prevent excessive exploration
    kappa = np.clip(kappa, 0.5, 5.0)
    
    ucb = -mu + kappa * sigma
    
    # Now blend EI and UCB
    # Use a weight that depends on how "confident" we are
    # When sigma is small relative to the gap, favor EI (exploitation)
    # When sigma is large, favor UCB (exploration)
    
    # Compute a local confidence measure
    # conf = 1 / (1 + sigma / (gap + eps))
    # But this varies per point. Instead, let's use a global blend.
    
    # Global uncertainty ratio
    if gap > 1e-10:
        unc_ratio = sigma_median / gap
    else:
        unc_ratio = 1.0
    
    # Weight for UCB: higher when uncertainty is high
    w_ucb = np.clip(unc_ratio, 0.0, 1.0)
    w_ei = 1.0 - w_ucb
    
    # Normalize EI and UCB to comparable scales
    # EI is in objective units, UCB is also in objective units (negative)
    # But they have different magnitudes. Let's use a weighted sum.
    
    scores = w_ei * ei + w_ucb * ucb
    
    # Safety: ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Time guard: if we've used too much time, return immediately
    elapsed = time.time() - start_time
    if elapsed > _DEADLINE_MARGIN:
        # This shouldn't happen for this simple computation, but just in case
        pass
    
    return scores

# EVOLVE-BLOCK-END
