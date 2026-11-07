import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    # Deadline guard
    t0 = time.time()
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Expected Improvement (EI) - standard
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
    
    # 2. Expected Improvement with exploration bonus (EI-EB)
    # Add a small bonus proportional to sigma to encourage exploration
    ei_eb = ei + 0.1 * sigma
    
    # 3. UCB with adaptive kappa based on number of evaluations
    # Since we don't know the exact step, use a moderate kappa
    ucb = -mu + 1.5 * sigma
    
    # 4. Probability of Improvement (PI)
    z_pi = (f_best - mu) / sigma_safe
    pi = norm.cdf(z_pi)
    
    # 5. Pure exploitation: -mu
    explo = -mu
    
    # 6. Thompson Sampling-like score: sample from posterior
    # Use a deterministic approximation: mu - sigma (lower quantile)
    ts_approx = mu - 1.0 * sigma
    
    # Collect scores
    score_list = [ei, ei_eb, ucb, pi, explo, ts_approx]
    
    # Normalize each to [0, 1]
    norm_scores = []
    for s in score_list:
        s_min = s.min()
        s_max = s.max()
        rng = s_max - s_min
        if rng < 1e-12:
            norm_scores.append(np.full_like(s, 0.5))
        else:
            norm_scores.append((s - s_min) / rng)
    
    # Weighted average: favor EI variants and UCB
    weights = np.array([0.25, 0.25, 0.2, 0.1, 0.1, 0.1])
    combined = np.average(norm_scores, axis=0, weights=weights)
    
    # Safety: ensure finite
    if not np.all(np.isfinite(combined)):
        combined = ei  # fallback
    
    # Check time (should be fast)
    if time.time() - t0 > 280:
        pass
    
    return combined
