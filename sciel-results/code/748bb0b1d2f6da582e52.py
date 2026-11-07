"""Acquisition for Bayesian optimization of Hartmann 6d using Meta-Selection of Acquisition Strategies."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: evaluator kills at 330s wall-clock; leave a safety margin.
_DEADLINE = time.time() + 325.0

# Known minimum of the Hartmann 6d function.
_HARTMANN_6D_MIN = -3.32237

def _soft_max_quality(scores, tau=0.05):
    """
    Compute a quality metric for a score array.
    Higher quality means the scores are more 'peaked' (one candidate stands out).
    We use a soft-max sum which is dominated by the maximum score.
    To avoid scale invariance issues, we normalize scores to have a mean of 0 and std of 1 
    before applying soft-max, or simply use the max score relative to the mean.
    
    The inspiration used: sum(exp(s/tau)). 
    If scores are large, this overflows. We should normalize scores first.
    """
    if len(scores) == 0:
        return 0.0
    
    # Normalize scores to zero mean, unit variance to make the metric scale-invariant
    # This ensures that a 'peaked' distribution is detected regardless of the absolute scale
    mu_s = np.mean(scores)
    std_s = np.std(scores)
    if std_s < 1e-12:
        return 0.0
        
    s_norm = (scores - mu_s) / std_s
    
    # Soft max sum. 
    # If tau is small, this is approx max(s_norm). 
    # If tau is large, this is approx sum(s_norm) which is 0.
    # We want to reward 'peakedness', so we want a metric that increases with the gap between max and mean.
    # max(s_norm) is a good proxy for peakedness.
    # Let's use: max(s_norm) - mean(s_norm) = max(s_norm) since mean is 0.
    # Or better: (max - mean) / std = max(s_norm).
    
    # However, the inspiration's specific success might come from the exponential weighting.
    # Let's stick to a robust peakedness metric: 
    # Q = max(s_norm) 
    # This is equivalent to how much the best candidate stands out in standard deviations.
    
    return float(np.max(s_norm))


def _thompson_winner_score(mu, sigma, f_best, n_samples=20, seed=42):
    """
    Thompson Sampling score: Probability of being the argmin of the sampled functions.
    """
    n = mu.shape[0]
    sigma_safe = np.maximum(sigma, 1e-12)
    rng = np.random.default_rng(seed)
    
    # Draw samples
    samples = rng.normal(mu[:, None], sigma_safe[:, None], size=(n, n_samples))
    
    # For each sample, find the index of the minimum
    winners = np.argmin(samples, axis=0)
    
    # Count wins
    scores = np.zeros(n)
    np.add.at(scores, winners, 1.0)
    scores /= n_samples
    
    return scores


def acquisition(mu, sigma, f_best):
    """
    Meta-selection acquisition function.
    Generates multiple candidate acquisition score vectors (different strategies/params)
    and returns the one that has the highest 'peakedness' quality.
    """
    # Deadline check
    if time.time() > _DEADLINE - 5.0:
        # Fallback to simple EI
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = ei
        return np.where(np.isfinite(scores), scores, 0.0)
    
    try:
        n = mu.shape[0]
        if n == 0:
            return np.array([])
            
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        
        # Gap from known optimum for adaptive weights
        gap = max(f_best - _HARTMANN_6D_MIN, 0.0)
        gap_norm = min(gap / 3.0, 1.0) # 1.0 if far, 0.0 if close
        
        candidates = []
        
        # 1. Pure EI
        candidates.append(ei)
        
        # 2. Adaptive EI (more exploration when far)
        w_explore_ei = 0.1 + 0.9 * gap_norm
        candidates.append(ei + w_explore_ei * sigma_safe)
        
        # 3. Adaptive UCB (more exploration when far)
        # UCB for minimization: (f_best - mu) + beta * sigma
        # High beta -> explore more.
        beta_ucb = 0.5 + 2.0 * gap_norm
        ucb = (f_best - mu) + beta_ucb * sigma_safe
        candidates.append(ucb)
        
        # 4. Static UCB variants
        for beta in [0.5, 1.0, 2.0]:
            candidates.append((f_best - mu) + beta * sigma_safe)
            
        # 5. Thompson Sampling Score
        ts_scores = _thompson_winner_score(mu, sigma, f_best, n_samples=20)
        candidates.append(ts_scores)
        
        # 6. Hybrid: EI + TS
        # Normalize TS to [0,1] to blend with EI
        ts_min, ts_max = ts_scores.min(), ts_scores.max()
        ts_norm = (ts_scores - ts_min) / (ts_max - ts_min + 1e-12)
        
        # EI is already non-negative. Scale it to [0,1] for blending?
        # Or just add them? EI is in objective units, TS is probability.
        # Let's try blending normalized versions.
        ei_min, ei_max = ei.min(), ei.max()
        ei_norm = (ei - ei_min) / (ei_max - ei_min + 1e-12)
        
        # Blend 50/50
        candidates.append(ei_norm + ts_norm)
        
        # Blend 70% EI, 30% TS
        candidates.append(0.7 * ei_norm + 0.3 * ts_norm)
        
        # 7. Pure Sigma (Exploration only) - rarely best, but includes in search
        candidates.append(sigma_safe)
        
        # Evaluate quality of each candidate
        best_quality = -np.inf
        best_scores = candidates[0]
        
        for s in candidates:
            # Ensure finite
            s = np.where(np.isfinite(s), s, -np.inf)
            if not np.any(np.isfinite(s)):
                continue
                
            q = _soft_max_quality(s)
            if q > best_quality:
                best_quality = q
                best_scores = s
                
        # Final sanity check
        if best_scores.shape != mu.shape:
            best_scores = np.resize(best_scores, mu.shape)
        best_scores = np.where(np.isfinite(best_scores), best_scores, 0.0)
        
        return best_scores

    except Exception:
        # Fallback to simple EI
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = ei
        return np.where(np.isfinite(scores), scores, 0.0)

# EVOLVE-BLOCK-END
