# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_start_time = time.time()
_DEADLINE = 330.0 # 300s limit + margin, but task says 300s per eval, total 330s wall clock kill.
# Actually, the task says "Each evaluation must finish within 300 s".
# And "evaluator KILLS the program after 330s wall-clock".
# Since acquisition is called 25 times, each call has 300s? No, "Each evaluation" usually means the whole loop step.
# But the contract says "For each seed... at each step... evaluates the candidate... until 30 evaluations".
# The "evaluation" in the context of the 300s limit likely refers to the entire optimization run or a single step?
# "Each evaluation must finish within 300 s" is ambiguous. Usually in these tasks, it means the function call or the step.
# Given 30 steps, if each step took 300s, total would be 9000s.
# It likely means the *entire* program run must finish in 330s.
# So I should track time in the module.

def acquisition(mu, sigma, f_best):
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 10: # Safety margin
        # Return a neutral score (e.g., based on mean) to avoid crashing
        return np.zeros_like(mu)

    # Handle numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # 1. Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Upper Confidence Bound (UCB) for minimization
    # We use a standard UCB with a fixed kappa for the component itself,
    # but the *weight* of this component will be adaptive.
    kappa_ucb = 2.0
    ucb = -mu + kappa_ucb * sigma_safe
    
    # Normalization to [0, 1]
    def min_max_normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx - mn < 1e-15:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    ei_norm = min_max_normalize(ei)
    ucb_norm = min_max_normalize(ucb)
    
    # 3. Adaptive Weighting
    # We want to explore (UCB) more when f_best is large (poor),
    # and exploit (EI) more when f_best is small (good).
    
    # Logistic function to map f_best to weight [0, 1]
    # Center c=10, Temperature T=5
    # If f_best >> 10, w -> 1 (High Exploration)
    # If f_best << 10, w -> 0 (High Exploitation)
    c = 10.0
    T = 5.0
    # Clip f_best to avoid overflow in exp
    f_safe = np.clip(f_best, -100, 100)
    w_explore = 1.0 / (1.0 + np.exp(-(f_safe - c) / T))
    
    # Ensure w_explore is within [0.1, 0.9] to always have some mix
    w_explore = np.clip(w_explore, 0.1, 0.9)
    
    # Combine
    score = w_explore * ucb_norm + (1.0 - w_explore) * ei_norm
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
