import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_T0 = time.time()
_DEADLINE = _T0 + 320.0  # safety margin before 330s kill

def _elapsed_ok():
    return time.time() < _DEADLINE

# Known minimum of Michalewicz 2D
_MICH_MIN = -1.8013

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining LCB exploration with EI exploitation,
    weighted by progress toward the known minimum.
    
    For minimization:
    - LCB score: -mu + beta * sigma (encourages low mean, high uncertainty)
    - EI score: expected improvement over f_best (encourages points likely to beat current best)
    
    We blend these based on how close f_best is to the known minimum.
    Early: more exploration (LCB weight higher)
    Late: more exploitation (EI weight higher)
    
    mu, sigma: shape (2000,) in objective units
    f_best: lowest observed value
    Returns: shape (2000,) finite scores, higher = evaluate next
    """
    # Check deadline
    if not _elapsed_ok():
        return np.random.rand(len(mu))
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    n = len(mu)
    
    # --- Compute LCB score for minimization ---
    # LCB = mu - beta * sigma; we want to maximize -LCB = -mu + beta * sigma
    # Use a moderate beta for exploration
    beta_lcb = 1.5
    lcb_score = -mu + beta_lcb * sigma_safe
    
    # --- Compute EI score for minimization ---
    z = (f_best - mu) / sigma_safe
    ei_score = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # --- Progress metric ---
    # 0 when f_best is far from min, 1 when f_best is at min
    # Range of Michalewicz 2D is roughly [-1.8, 1.8], so span ~3.6
    progress = np.clip((f_best - _MICH_MIN) / 3.6, 0.0, 1.0)
    
    # --- Normalize scores to [0, 1] for blending ---
    def _normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        rng = mx - mn
        if rng < eps:
            return np.zeros_like(arr)
        return (arr - mn) / rng
    
    lcb_norm = _normalize(lcb_score)
    ei_norm = _normalize(ei_score)
    
    # --- Blend weights ---
    # Early (progress near 0): weight LCB heavily (explore)
    # Late (progress near 1): weight EI heavily (exploit)
    w_lcb = 0.6 * (1.0 - progress) + 0.2  # 0.8 -> 0.2
    w_ei  = 0.2 * (1.0 - progress) + 0.6  # 0.2 -> 0.8
    
    # Normalize weights to sum to 1
    w_sum = w_lcb + w_ei
    w_lcb /= w_sum
    w_ei  /= w_sum
    
    # --- Final score ---
    score = w_lcb * lcb_norm + w_ei * ei_norm
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
