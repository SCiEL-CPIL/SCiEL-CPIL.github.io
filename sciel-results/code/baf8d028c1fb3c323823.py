import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
TIME_LIMIT = 330.0
SAFETY_MARGIN = 5.0
_eval_count = 0

def acquisition(mu, sigma, f_best):
    global _eval_count
    _eval_count += 1
    
    # Deadline guard: return safe scores if approaching time limit
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT - SAFETY_MARGIN:
        # If we are at the deadline, return a neutral score (e.g., just -mu)
        # to force a selection quickly.
        return -mu
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Thompson Sampling approach: sample from the posterior and pick the best
    # Sample from the GP posterior: x ~ N(mu, sigma^2)
    # We want to minimize, so we pick the candidate with the lowest sampled value
    # But since we can only return scores, we use a deterministic approximation:
    # Use -mu + sigma as a proxy for Thompson Sampling (mean + std)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute Probability of Improvement (PI) for minimization
    pi = norm.cdf(z)
    
    # Use a weighted combination that adapts to the number of evaluations
    # Early on: more exploration (weight PI and LCB)
    # Later: more exploitation (weight EI)
    
    # Adaptive beta for LCB
    beta = 1.0 + 2.0 * np.exp(-_eval_count / 8.0)
    
    # LCB for minimization
    lcb = -mu + beta * sigma_safe
    
    # Normalize each component to [0, 1]
    def normalize(arr):
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        arr_range = arr_max - arr_min + 1e-10
        return (arr - arr_min) / arr_range
    
    ei_norm = normalize(ei)
    pi_norm = normalize(pi)
    lcb_norm = normalize(lcb)
    
    # Adaptive weights based on evaluation count
    # Early: 50% LCB, 30% PI, 20% EI
    # Late: 20% LCB, 10% PI, 70% EI
    t = _eval_count / 30.0  # Normalize to [0, 1] over 30 evaluations
    t = np.clip(t, 0.0, 1.0)
    
    lcb_weight = 0.5 * (1 - t) + 0.2 * t
    pi_weight = 0.3 * (1 - t) + 0.1 * t
    ei_weight = 0.2 * (1 - t) + 0.7 * t
    
    # Ensure weights sum to 1
    total_weight = lcb_weight + pi_weight + ei_weight
    lcb_weight /= total_weight
    pi_weight /= total_weight
    ei_weight /= total_weight
    
    scores = lcb_weight * lcb_norm + pi_weight * pi_norm + ei_weight * ei_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
