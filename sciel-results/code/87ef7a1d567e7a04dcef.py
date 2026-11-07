# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = None
_deadline = None

def _elapsed():
    if _start_time is None:
        return 0.0
    return time.time() - _start_time

def _ei_min(mu, sigma, f_best):
    """Expected Improvement for minimization, clipped >= 0, vectorized."""
    eps = 1e-12
    s = np.maximum(sigma, eps)
    z = (f_best - mu) / s
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
    return np.maximum(ei, 0.0), s

def _pi_min(mu, sigma, f_best):
    """Probability of Improvement for minimization."""
    eps = 1e-12
    s = np.maximum(sigma, eps)
    z = (f_best - mu) / s
    pi = norm.cdf(z)
    return np.maximum(pi, 0.0)

def _ucb_score(mu, sigma, beta):
    """LCB-based score (higher = more promising for minimization): -mu + beta*sigma."""
    return -mu + beta * np.maximum(sigma, 1e-12)

def _kriging_believer_score(mu, sigma, f_best, cost, alpha=1.0):
    """
    Kriging Believer strategy:
    1. Compute EI for each candidate
    2. Find the EI value of the best candidate
    3. Convert to z-score: z* = (f_best - mu_best) / sigma_best
    4. Score each candidate by: mu(x) + sigma(x) * z*  (this is the "believer" mean)
       Actually, the Kriging Believer scores by the GP mean at the point that would be
       selected if we were doing a "believer" pass - but here we approximate by
       scoring each point by how likely it is to be the EI-maximizer.
    
    A simpler and more effective variant:
    - Compute EI scores
    - The "believer" point is argmax EI
    - For scoring, we want points that are likely to be near the believer point
    - Alternative: score = -mu + alpha * sigma * (some exploration term)
    
    Actually, the classic Kriging Believer for minimization:
    - The EI-maximizer is the point where (f_best - mu)/sigma is maximized (the z-score)
    - We can score points by their "believer value": if we were to pick the point with
      maximum z-score, what would its predicted value be?
    - For a score function: score(x) = -mu(x) + sigma(x) * z*_global
      where z*_global is the z-score of the current best EI candidate.
    
    This effectively creates a UCB-like score with an adaptive beta that matches
    the current EI landscape.
    """
    s = np.maximum(sigma, 1e-12)
    z_scores = (f_best - mu) / s
    # The z-score of the best EI candidate
    z_star = np.max(z_scores)
    
    # Kriging Believer score: points that achieve high z-scores are favored
    # Score = -mu + sigma * z_star (this is like UCB with beta = z_star)
    # But we also want to incorporate cost awareness
    
    # Cost-adjusted: prefer cheaper points among those with similar scores
    cost_eff = 1.0 / (cost + 1e-12)
    
    # Normalize components
    def norm_arr(a):
        lo, hi = np.min(a), np.max(a)
        rng = hi - lo
        if rng < 1e-12:
            return np.ones_like(a) * 0.5
        return (a - lo) / rng
    
    kb_raw = -mu + s * z_star
    kb_n = norm_arr(kb_raw)
    ce_n = norm_arr(cost_eff)
    
    # Weighted combination
    score = 0.85 * kb_n + 0.15 * ce_n
    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

def _adaptive_cost_aware_score(mu, sigma, f_best, cost):
    """
    Adaptive cost-aware acquisition that blends PI, EI, and cost efficiency.
    Uses the distribution of sigma to adapt exploration.
    """
    s = np.maximum(sigma, 1e-12)
    
    # Probability of Improvement
    pi = _pi_min(mu, sigma, f_best)
    
    # Expected Improvement
    ei, _ = _ei_min(mu, sigma, f_best)
    
    # Cost efficiency
    cost_eff = 1.0 / (cost + 1e-12)
    
    # Normalize
    def norm_arr(a):
        lo, hi = np.min(a), np.max(a)
        rng = hi - lo
        if rng < 1e-12:
            return np.ones_like(a) * 0.5
        return (a - lo) / rng
    
    pi_n = norm_arr(pi)
    ei_n = norm_arr(ei)
    ce_n = norm_arr(cost_eff)
    
    # Adaptive weights: when sigma is large (early), weight PI more; when sigma is small, weight EI more
    sigma_median = np.median(sigma)
    sigma_max = np.max(sigma)
    if sigma_max < 1e-12:
        w_pi, w_ei, w_cost = 0.4, 0.4, 0.2
    else:
        # Ratio of median to max sigma: high ratio means uniform uncertainty, low means concentrated
        ratio = sigma_median / sigma_max
        # When ratio is high, uncertainty is spread out -> PI is more useful
        # When ratio is low, uncertainty is concentrated -> EI targets the high-uncertainty regions
        w_pi = 0.3 + 0.3 * ratio
        w_ei = 0.3 + 0.3 * (1 - ratio)
        w_cost = 0.1
    
    score = w_pi * pi_n + w_ei * ei_n + w_cost * ce_n
    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Kriging Believer strategy.
    Returns 2000 finite scores (higher = evaluate next).
    """
    global _start_time, _deadline
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 320.0  # safety margin under 330s kill
    
    if time.time() > _deadline:
        # Emergency: cheap cost-aware PI, always finite
        pi = _pi_min(mu, sigma, f_best)
        ce = 1.0 / (cost + 1e-12)
        out = 0.7 * pi + 0.3 * ce
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Primary strategy: Kriging Believer with cost awareness
    try:
        score = _kriging_believer_score(mu, sigma, f_best, cost)
    except Exception:
        # Fallback to adaptive cost-aware score
        try:
            score = _adaptive_cost_aware_score(mu, sigma, f_best, cost)
        except Exception:
            # Last resort: pure PI with cost
            pi = _pi_min(mu, sigma, f_best)
            ce = 1.0 / (cost + 1e-12)
            score = 0.7 * pi + 0.3 * ce
    
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure all values are finite and non-negative
    score = np.maximum(score, 0.0)
    
    # Scale so max is 1 for stability
    mx = np.max(score)
    if mx > 1e-12:
        score = score / mx
    
    return score

# EVOLVE-BLOCK-END
