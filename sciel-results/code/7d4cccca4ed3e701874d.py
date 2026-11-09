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
    
    # Clip z to avoid numerical issues with norm.cdf and norm.pdf
    z = np.clip(z, -10, 10)
    
    # Standard Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Use Upper Confidence Bound (UCB) for better exploration-exploitation balance
    # UCB = -mu + beta * sigma (minimization problem)
    # The finding shows UCB with explicit balance outperforms pure EI in low-budget regime
    # We combine EI and UCB: use UCB as the primary scoring mechanism
    # beta is chosen to balance exploration; for 2D Ackley with [-5,5]^2, 
    # a moderate beta works well. We use a time-adaptive beta that starts
    # with more exploration and decreases over time.
    
    # Estimate progress: if f_best is very small, we're close to optimum,
    # so reduce exploration. Otherwise, maintain exploration.
    # For Ackley, the minimum is 0, so f_best approaching 0 means we're done.
    # Use a simple heuristic: beta decreases as f_best gets closer to 0.
    # At start, f_best might be ~2-3, at end it might be ~0.001.
    
    # Adaptive beta: start with higher exploration, decrease as we converge
    # f_best near 0 means we're close to optimum, so reduce exploration
    # f_best large means we need more exploration
    # Scale beta based on f_best: when f_best is large, beta is larger
    # When f_best is small, beta is smaller
    # Use a smooth transition: beta approaches beta_min as f_best -> 0
    # beta approaches beta_max as f_best -> infinity
    beta_min = 0.3
    beta_max = 1.0
    scale = 0.5  # Controls how quickly beta decreases as f_best approaches 0
    
    # Smooth adaptation using exponential decay
    beta = beta_min + (beta_max - beta_min) * np.exp(-f_best / scale)
    
    ucb = -mu + beta * sigma_safe
    
    # Normalize EI to [0, 1] range relative to f_best for blending
    # If f_best is very small, EI is already small, so normalization is safe
    # Use a safe denominator to avoid division by zero
    denom = max(f_best, 1e-6)
    ei_normalized = ei / denom
    
    # Blend: UCB as primary, EI as secondary
    # When f_best is large (early on), weight UCB more for exploration
    # When f_best is small (late on), weight EI more for exploitation
    # Adaptive blending: w_ucb decreases as f_best decreases
    # w_ucb = 0.7 when f_best is large, w_ucb = 0.4 when f_best is small
    w_ucb = 0.4 + 0.3 * (f_best / (f_best + scale))
    w_ei = 1.0 - w_ucb
    
    score = w_ucb * ucb + w_ei * ei_normalized
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
