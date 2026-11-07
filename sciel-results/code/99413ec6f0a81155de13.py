import numpy as np
from scipy.stats import norm
import time

# Deadline guard: evaluator kills after 330s wall-clock.
# We track time and bail out early if needed.
_DEADLINE = 320.0  # safety margin
_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Dynamic acquisition for Bayesian optimization of sphere 1d.
    
    Strategy: Blend Expected Improvement (EI) with a quantile-based exploration
    term. The blend is adaptive: when uncertainty is high relative to the
    improvement scale, we weight exploration more; when we're close to the
    optimum, we exploit more.
    
    For minimization:
      EI(x) = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
      where Phi is the standard normal CDF and phi is the standard normal PDF.
    
    We use a hybrid score:
      score = w_ei * EI + w_explore * (z_alpha * sigma - mu)
    
    The weights are determined by the current state:
      - If sigma is large relative to (f_best - min_mu), we're in exploration mode
      - If sigma is small, we're in exploitation mode
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Return a simple LCB as fallback
        score = -mu + 2.0 * sigma
        score = np.where(np.isfinite(score), score, -1e18)
        return score
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-12)
    
    # Compute EI for minimization
    # Improvement = f_best - mu (positive when mu < f_best)
    improvement = f_best - mu
    # Standardize
    z = improvement / sigma
    # EI = improvement * Phi(z) + sigma * phi(z)
    # For large negative z (mu >> f_best), EI ~ 0
    # For large positive z (mu << f_best), EI ~ improvement
    
    # Numerically stable computation
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    EI = improvement * Phi_z + sigma * phi_z
    EI = np.maximum(EI, 0.0)
    
    # Exploration term: lower quantile (for minimization, we want low values)
    # Use a dynamic quantile that depends on the state
    # When we're far from optimum, explore more (lower quantile = more aggressive)
    # When close, exploit more (higher quantile = more conservative)
    
    # Measure of how "close" we are: ratio of typical uncertainty to best improvement
    min_mu = np.min(mu)
    typical_sigma = np.median(sigma)
    
    # If f_best is much larger than min_mu, we haven't found a good point yet
    # If f_best is close to min_mu, we're exploiting
    gap = f_best - min_mu
    if gap > 1e-10:
        # Relative uncertainty
        rel_uncert = typical_sigma / max(gap, 1e-10)
    else:
        rel_uncert = 1.0
    
    # Dynamic alpha: when uncertain, use a more aggressive quantile (smaller alpha)
    # alpha ranges from 1e-4 (very exploratory) to 0.05 (more exploitative)
    # Map rel_uncert to alpha
    if rel_uncert > 5.0:
        alpha = 1e-4
    elif rel_uncert < 0.1:
        alpha = 0.05
    else:
        alpha = 1e-4 * (rel_uncert / 5.0) + 0.05 * (1.0 - rel_uncert / 5.0)
        alpha = np.clip(alpha, 1e-4, 0.05)
    
    z_alpha = norm.ppf(1 - alpha)
    
    # Quantile-based exploration score (for minimization: -mu + z_alpha * sigma)
    explore_score = -mu + z_alpha * sigma
    
    # Adaptive weighting
    # When EI is large, trust EI more
    # When EI is small (we're near the optimum), use exploration to escape local optima
    ei_scale = np.mean(EI) + 1e-10
    ei_weight = np.clip(ei_scale / (typical_sigma + 1e-10), 0.3, 1.0)
    explore_weight = 1.0 - ei_weight
    
    # Combine
    score = ei_weight * EI + explore_weight * explore_score
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -1e18)
    
    return score
