import numpy as np
from scipy.stats import norm
import time

# Global variable to track start time for deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 285.0  # Safety margin before 300s

def acquisition(mu, sigma, f_best):
    # Deadline guard: If we are very close to the time limit,
    # return a score that strongly favors exploitation (low mu)
    # to avoid wasting time on exploration or complex calculations.
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback to pure exploitation: lower mu is better
        # Invert mu so higher score = lower mu
        return -mu
    
    # Ensure sigma is safe for division
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    # Standard Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Thompson Sampling heuristic: score = -mu + gamma * sigma
    # where gamma is a dynamic, quantile-normalized exploration weight.
    # The exploration weight is scaled by the current best value to aggressively
    # target high-variance regions early and sharply focus on the global minimum
    # as f_best approaches zero.
    
    # Compute the quantile-normalized sigma: rank sigma values and normalize to [0, 1]
    # This ensures that the exploration term is relative to the uncertainty landscape
    sigma_rank = np.argsort(np.argsort(sigma_safe))
    sigma_quantile = sigma_rank / (len(sigma_safe) - 1)  # Normalize to [0, 1]
    
    # Adaptive exploration weight:
    # When f_best is large (early), we want more exploration -> higher gamma
    # When f_best is small (late), we want less exploration -> lower gamma
    # Use a smooth transition: gamma_base = gamma_min + (gamma_max - gamma_min) * (f_best / (f_best + scale))
    gamma_min = 0.3
    gamma_max = 2.0
    scale = 0.5
    gamma_base = gamma_min + (gamma_max - gamma_min) * (f_best / (f_best + scale))
    
    # The exploration term: gamma_base * sigma_quantile * sigma_safe
    # This scales the uncertainty by both the adaptive weight and the relative rank of uncertainty
    exploration_term = gamma_base * sigma_quantile * sigma_safe
    
    # Thompson Sampling score: -mu + exploration_term
    ts_score = -mu + exploration_term
    
    # Blend EI and Thompson Sampling:
    # When f_best is large (early), weight Thompson Sampling more for exploration
    # When f_best is small (late), weight EI more for exploitation
    # Adaptive blending: w_ts decreases as f_best decreases
    # w_ts = 0.7 when f_best is large, w_ts = 0.3 when f_best is small
    w_ts = 0.3 + 0.4 * (f_best / (f_best + scale))
    w_ei = 1.0 - w_ts
    
    # Normalize EI to comparable scale
    denom = max(f_best, 1e-6)
    ei_normalized = ei / denom
    
    # Blend: Thompson Sampling as primary, EI as secondary
    score = w_ts * ts_score + w_ei * ei_normalized
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
