# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 330.0
_call_count = 0

def _logistic(x):
    """Numerically stable sigmoid."""
    # clip to avoid overflow
    x = np.clip(x, -500, 500)
    out = np.zeros_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out

def acquisition(mu, sigma, f_best):
    global _call_count
    _call_count += 1
    n = mu.size

    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 20:
        return np.zeros(n, dtype=float)

    # Sanitize inputs
    sigma = np.asarray(sigma, dtype=float)
    mu = np.asarray(mu, dtype=float)
    f_best = float(f_best)
    sigma_safe = np.maximum(sigma, 1e-12)

    # 1. Expected Improvement (for minimization)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # 2. Lower Confidence Bound (LCB) for minimization
    # We want to explore areas with high uncertainty (high sigma) and low mean.
    # LCB = mu - kappa * sigma
    # Higher score means lower LCB.
    # We adapt kappa based on the progress (f_best).
    # Early on (high f_best), we want more exploration (larger kappa).
    # Late on (low f_best), we want more exploitation (smaller kappa).
    
    # A simple heuristic: 
    # Beale function minimum is 0.
    # Typical initial random values are large (e.g., 10-100).
    # Let's normalize f_best. 
    # If f_best is large, kappa should be large.
    # If f_best is small, kappa should be small.
    
    # Use a scaling that drops off as f_best decreases.
    # kappa = base_kappa * (1 + f_best / scale)
    # But we want to ensure we don't over-explore if we are close to 0.
    
    # Let's try a different structural approach: 
    # Instead of just EI or LCB, use a hybrid that is robust.
    # The previous attempt used min-max normalization which can be unstable if the distribution is narrow.
    # Let's use a soft normalization or direct combination.
    
    # Strategy: 
    # Score = alpha * EI + (1 - alpha) * (-LCB_normalized)
    # Actually, EI is already a good measure of expected gain.
    # LCB is good for exploration.
    
    # Let's look at the distribution of EI.
    # If we just return EI, it is a very strong baseline.
    # The score 2.06 suggests the current mixed approach is okay but not great.
    # The inspiration 1 (score 6.79) might be using pure EI or a very specific UCB.
    # Let's try to improve EI by adding a small exploration term that decays.
    
    # Decay factor based on number of calls? We have access to _call_count.
    # But f_best is more informative.
    
    # Let's use: Score = EI + gamma * sigma
    # where gamma is small. This encourages exploring high uncertainty areas 
    # even if EI is low, but EI dominates when improvement is likely.
    
    # How to choose gamma?
    # If f_best is high, we can afford more exploration.
    # If f_best is low, we should stick to EI.
    
    # Let's define gamma based on f_best.
    # gamma = 0.1 * np.exp(-f_best / 5.0)  # Decays fast as we get good
    # Or maybe a smoother decay.
    
    # Alternative: Use a "Noisy EI" or "Thompson Sampling" approximation?
    # Thompson Sampling would require sampling from the posterior, which we can't do here 
    # (we only have mu and sigma, not the full covariance).
    # So we are stuck with point-based acquisition.
    
    # Let's refine the EI + UCB mix.
    # The issue with min-max normalization is that if one point is an outlier, 
    # it squashes the rest. 
    # Let's use a standard deviation based normalization.
    
    def robust_normalize(arr):
        mn = np.median(arr)
        std = np.std(arr)
        if std < 1e-12:
            return np.zeros_like(arr)
        z_scores = (arr - mn) / std
        # Map to [0, 1] via sigmoid or similar? 
        # Or just return z_scores and rely on the relative magnitudes?
        # The evaluator picks the argmax. So relative magnitude is what matters.
        # We don't need to normalize to [0,1] strictly, just ensure they are on comparable scales.
        return z_scores
    
    ei_z = robust_normalize(ei)
    # For UCB (exploration), we use sigma. 
    # Actually, -mu + kappa*sigma is the LCB score.
    # Let's combine the raw EI and raw LCB scores.
    
    # Let's try: Score = EI + beta * sigma
    # beta controls exploration.
    # We want beta to be large when f_best is large.
    # beta = 0.5 * (f_best + 1) / (f_best + 10) ?
    # If f_best = 100, beta ~ 0.45
    # If f_best = 1, beta ~ 0.045
    # If f_best = 0, beta = 0.05
    
    # Let's test this structure.
    beta = 0.5 * (f_best + 1.0) / (f_best + 10.0)
    
    # Also, pure EI can be very peaked. 
    # Sometimes we want to look at areas where sigma is high even if mu is not great.
    # The term beta * sigma does this.
    
    score = ei + beta * sigma_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
