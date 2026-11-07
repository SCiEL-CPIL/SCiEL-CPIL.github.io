"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track when acquisition was first called to enforce a global deadline
_acq_start_time = [None]
_DEADLINE_SECONDS = 320.0  # Safety margin before 330s kill


def _get_elapsed():
    """Return elapsed seconds since first acquisition call, or 0 if not started."""
    if _acq_start_time[0] is None:
        return 0.0
    return time.time() - _acq_start_time[0]


def _ensure_started():
    """Record start time on first call."""
    if _acq_start_time[0] is None:
        _acq_start_time[0] = time.time()


def _ei_scores(mu, sigma, f_best):
    """Compute Expected Improvement scores."""
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    return ei


def _lcb_scores(mu, sigma, f_best, xi=2.0):
    """Compute Lower Confidence Bound scores (higher is better)."""
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    # LCB for minimization: - (mu - xi * sigma) = xi*sigma - mu
    lcb = xi * sigma_safe - mu
    lcb = np.where(np.isfinite(lcb), lcb, 0.0)
    return lcb


def _quantile_expected_min(mu, sigma, f_best, k=5):
    """
    Quantile-based expected minimum: estimate E[min of k samples from N(mu, sigma^2)]
    using the k-th order statistic quantile approximation.
    
    For a standard normal, the expected minimum of k samples is approximately
    -E[max of k standard normals]. We use a simple approximation:
    E[min_k] ≈ mu - sigma * q where q depends on k.
    
    For k=5, the expected minimum of 5 standard normals ≈ -1.16
    For k=10, ≈ -1.54
    For k=20, ≈ -1.88
    
    We adapt k based on how uncertain we are (larger sigma → larger k for more exploration).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive k: more samples where uncertainty is high
    # Normalize sigma by typical scale
    sigma_mean = np.mean(sigma_safe)
    sigma_max = np.max(sigma_safe)
    
    if sigma_max > 1e-8:
        # Scale factor: 1.0 to 3.0 based on relative uncertainty
        scale = 1.0 + 2.0 * (sigma_safe / sigma_max)
    else:
        scale = np.ones_like(sigma_safe)
    
    # Use a fixed effective k that balances exploration/exploitation
    # k_eff ranges from ~3 (exploit) to ~15 (explore)
    k_eff = 3.0 + 12.0 * scale / np.max(scale) if np.max(scale) > 0 else 3.0
    
    # Precompute quantile factors for a range of k values
    # E[min of k iid N(0,1)] ≈ -c(k)
    # Using approximation: c(k) ≈ 0.5772 * ln(k) + 0.5772 (Euler-Mascheroni based)
    # Better: use known values
    # k=1: 0, k=2: 0.564, k=3: 0.846, k=5: 1.163, k=10: 1.539, k=20: 1.880, k=50: 2.251
    # Linear interpolation on log scale
    k_table = np.array([1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0])
    c_table = np.array([0.0, 0.564, 0.846, 1.163, 1.539, 1.880, 2.251])
    
    # Interpolate c(k_eff)
    c_vals = np.interp(k_eff, k_table, c_table)
    
    # Expected minimum of k samples from N(mu, sigma^2)
    # E[min] ≈ mu - sigma * c(k)
    expected_min = mu - sigma_safe * c_vals
    
    # Score: how much better is this than f_best?
    # We want higher scores for points likely to have low values
    # Use the negative of expected minimum relative to f_best
    score = f_best - expected_min
    score = np.where(np.isfinite(score), score, 0.0)
    return score


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining EI, LCB, and quantile-based expected minimum.
    
    Uses adaptive weighting based on elapsed time (proxy for number of evaluations):
    - Early: more exploration (LCB + quantile)
    - Late: more exploitation (EI)
    """
    _ensure_started()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    elapsed = _get_elapsed()
    
    # Check deadline
    if elapsed > _DEADLINE_SECONDS:
        # Return simple EI as fallback
        return _ei_scores(mu, sigma, f_best)
    
    # Compute all three scores
    ei = _ei_scores(mu, sigma, f_best)
    
    # Adaptive LCB xi: start high, decrease over time
    # elapsed ≈ 0 to ~320s, 25 evaluations (30 total - 5 initial)
    # Roughly 12.8s per evaluation
    progress = min(elapsed / 250.0, 1.0)  # 0 at start, 1 near end
    xi = 3.0 - 2.0 * progress  # xi from 3.0 to 1.0
    
    lcb = _lcb_scores(mu, sigma, f_best, xi=xi)
    
    qem = _quantile_expected_min(mu, sigma, f_best)
    
    # Normalize each score to [0, 1] range for blending
    def _normalize(s):
        s_min = np.min(s)
        s_max = np.max(s)
        if s_max - s_min > 1e-12:
            return (s - s_min) / (s_max - s_min)
        return np.zeros_like(s)
    
    ei_norm = _normalize(ei)
    lcb_norm = _normalize(lcb)
    qem_norm = _normalize(qem)
    
    # Adaptive weights: exploration early, exploitation late
    w_ei = 0.3 + 0.5 * progress       # 0.3 → 0.8
    w_lcb = 0.4 - 0.2 * progress      # 0.4 → 0.2
    w_qem = 0.3 - 0.1 * progress      # 0.3 → 0.2
    
    # Normalize weights to sum to 1
    w_sum = w_ei + w_lcb + w_qem
    w_ei /= w_sum
    w_lcb /= w_sum
    w_qem /= w_sum
    
    # Blend
    combined = w_ei * ei_norm + w_lcb * lcb_norm + w_qem * qem_norm
    
    # Ensure finite
    combined = np.where(np.isfinite(combined), combined, 0.0)
    
    return combined
# EVOLVE-BLOCK-END
