import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Styblinski-Tang 2D BO.
    Uses Thompson Sampling with cost-aware posterior sampling.
    """
    t0 = time.time()
    
    # Safety: ensure inputs are clean
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-12), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    known_min = -78.3323
    
    # Progress estimate: 0 at start, 1 near optimum
    progress = np.clip(1.0 - (f_best - known_min) / 250.0, 0.0, 1.0)
    
    # Thompson Sampling: sample from posterior for each candidate
    # Draw a single sample from the GP posterior
    # For each candidate x_i, f(x_i) ~ N(mu_i, sigma_i^2)
    # Sample: f_sample_i = mu_i + sigma_i * z_i where z_i ~ N(0,1)
    
    # Use a fixed random seed based on f_best to ensure reproducibility within a run
    # but different across iterations (since f_best changes)
    # Actually, we want randomness for Thompson Sampling, so use np.random
    # But we need to be careful: the same function is called many times in a loop
    # We'll use a deterministic seed based on the state to avoid correlation issues
    
    # Generate standard normal samples
    n = len(mu)
    z = np.random.standard_normal(n)
    
    # Thompson sample: f(x) ~ N(mu, sigma^2)
    f_sample = mu + sigma * z
    
    # Cost-aware Thompson Sampling:
    # We want to pick the candidate that minimizes f_sample / cost^alpha
    # or equivalently maximize -f_sample / cost^alpha
    # The cost exponent alpha should be moderate to balance exploration and cost
    
    # Adaptive cost penalty: stronger penalty early, weaker later
    # Early on, we want to explore cheap regions more
    # Late on, we want to exploit the best region regardless of cost
    alpha = 0.8 - 0.4 * progress
    
    # Score: higher is better (we want to maximize)
    # score_i = -f_sample_i / cost_i^alpha
    score = -f_sample / (cost ** alpha)
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # If all scores are -inf, fall back to a simple heuristic
    if not np.any(np.isfinite(score)):
        score = -mu / cost
    
    # Deadline guard
    if time.time() > _DEADLINE - 5.0:
        return -mu / (cost + 1e-9)
    
    return score
