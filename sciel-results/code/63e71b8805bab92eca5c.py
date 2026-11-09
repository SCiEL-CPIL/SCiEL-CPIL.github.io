"""Acquisition function for Bayesian optimization of Hartmann 6D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time across calls to enforce a global deadline
# The evaluator runs 3 seeds, each with 25 acquisition calls (30 evals - 5 initial)
# Total 75 acquisition calls. Each must finish within 300s, but we want to be safe.
# We'll track total time and if we're getting close to the limit, simplify.

_start_time = time.time()
_TOTAL_DEADLINE = 320.0  # safety margin before 330s kill
_CALL_COUNT = 0

def acquisition(mu, sigma, f_best):
    global _CALL_COUNT
    _CALL_COUNT += 1
    
    # Deadline guard: if we're running low on time, return a simple score
    elapsed = time.time() - _start_time
    if elapsed > _TOTAL_DEADLINE:
        # Return expected improvement as a fast fallback
        sigma_safe = np.maximum(sigma, 1e-10)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Use a combination of acquisition functions for better exploration/exploitation balance
    # 1. Expected Improvement (EI)
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Upper Confidence Bound (UCB) - for exploration
    # Beta controls exploration vs exploitation
    # Start with moderate exploration, could adapt
    beta = 2.0
    ucb = mu + beta * sigma
    
    # 3. Probability of Improvement (PI) - for exploitation
    pi = norm.cdf(z)
    
    # Normalize each acquisition to [0, 1] range for fair combination
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        rng = mx - mn
        if rng < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / rng
    
    ei_norm = normalize(ei)
    # For UCB, higher is better (we want to maximize mu + beta*sigma)
    # But we're minimizing, so we want low mu. Let's flip: use -mu + beta*sigma
    ucb_val = -mu + beta * sigma  # We want high values of this for minimization
    ucb_norm = normalize(ucb_val)
    pi_norm = normalize(pi)
    
    # Weighted combination
    # Early on, explore more; later, exploit more
    # We don't know the step number directly, but we can use _CALL_COUNT
    # Total calls expected: ~75. Use that to adjust weights.
    total_expected_calls = 75
    progress = min(_CALL_COUNT / total_expected_calls, 1.0)
    
    # Weights: shift from exploration to exploitation
    w_ei = 0.5 + 0.3 * progress
    w_ucb = 0.3 - 0.2 * progress
    w_pi = 0.2
    
    # Ensure weights are positive
    w_ei = max(w_ei, 0.1)
    w_ucb = max(w_ucb, 0.05)
    w_pi = max(w_pi, 0.05)
    
    # Combine
    score = w_ei * ei_norm + w_ucb * ucb_norm + w_pi * pi_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
