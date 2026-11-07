# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_deadline = None
_call_count = 0
_last_mu_mean = None
_last_sigma_mean = None

def acquisition(mu, sigma, f_best, cost):
    """
    Compute an acquisition score for each candidate.
    Returns a (2000,) array of finite scores (higher = more preferred).
    """
    global _deadline, _call_count, _last_mu_mean, _last_sigma_mean
    start_time = time.time()
    _call_count += 1
    
    eps = 1e-10
    n = len(mu)
    
    # Safety: 330s wall-clock total.
    # If we are running out of time, return a simple, fast fallback.
    # However, the calculation below is O(N) and very fast (~1ms), so this is rarely triggered
    # unless the system is extremely slow or under load.
    
    sigma_safe = np.maximum(sigma, eps)
    
    # 1. Expected Improvement (EI)
    # For minimization: EI = E[max(f_best - f, 0)]
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Handle division by zero safely (already done via sigma_safe)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # 2. Probability of Improvement (PI)
    # PI = P(f < f_best) = Phi(z)
    pi = norm.cdf(z)
    
    # 3. Lower Confidence Bound (LCB) / UCB score
    # We want to minimize f. LCB = mu - beta * sigma.
    # Score for selection: -LCB = -mu + beta * sigma.
    # We need to choose beta.
    # A common choice is beta = log(t + 1) or a constant.
    # In high dimensions, we often need higher exploration.
    # Let's use a dynamic beta that increases slightly with the number of calls,
    # but capped to avoid excessive noise.
    # t = _call_count
    # beta = 1.0 + 0.5 * np.log(t + 1)  # Maybe too complex?
    
    # Let's stick to a robust fixed beta for stability, or a simple dynamic one.
    # The previous successful run used beta=1.5.
    # Let's try to make it slightly adaptive based on the observed sigma magnitude.
    # If sigma is generally small, we can be more exploitative.
    # If sigma is large, we need more exploration.
    
    # Let's calculate a scale factor based on the current sigma distribution.
    # sigma_median = np.median(sigma)
    # beta = 1.5 * (sigma_median / (sigma_median + 1e-6)) # This is basically 1.5 if sigma is not tiny.
    
    # Let's try a "Hybrid" score that is more robust.
    # Instead of just linear combination, let's consider the "Cost-Aware EI" more carefully.
    
    # Strategy:
    # The cost is 1 + mean(x).
    # We want to maximize (Score / Cost).
    # Score should reflect the expected gain in finding a better minimum.
    
    # Let's refine the combination.
    # Standardization is good but can be unstable if variance is very small.
    # Let's use a robust standardization or just normalize by a fixed scale if possible.
    # However, since mu and sigma are in objective units, and the objective scale is known
    # (Hartmann 6D min is ~-3.3, max is ~3.0), we can potentially normalize EI and UCB
    # by a fixed range, e.g., 6.0 (approx range of function values).
    
    # Let's try normalizing by the range of the function values observed so far?
    # We don't have the history of f values, only f_best.
    # We can assume a typical scale.
    
    # Let's stick to per-call standardization as it is adaptive to the GP's state.
    # But we'll use a safer standardization: (x - mean) / (std + eps)
    
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + eps
    ei_norm = (ei - ei_mean) / ei_std
    
    pi_mean = np.mean(pi)
    pi_std = np.std(pi) + eps
    pi_norm = (pi - pi_mean) / pi_std
    
    beta = 1.5
    ucb_score = -mu + beta * sigma_safe
    
    ucb_mean = np.mean(ucb_score)
    ucb_std = np.std(ucb_score) + eps
    ucb_norm = (ucb_score - ucb_mean) / ucb_std
    
    # Weights
    # The previous best (1.027) used w_ei=0.6, w_pi=0.1, w_ucb=0.3.
    # To make a distinct candidate, we should change the strategy slightly.
    # Perhaps we can use a different combination or a different cost penalty.
    
    # Idea: Use "Expected Improvement per unit cost" more directly.
    # Instead of dividing the final score by cost, we can incorporate cost into the EI calculation?
    # Or we can use a different weight distribution.
    
    # Let's try increasing exploration slightly, as Hartmann 6D is multi-modal.
    # w_ei = 0.5
    # w_pi = 0.0
    # w_ucb = 0.5
    # And use a slightly stronger cost penalty: 1 / (cost ** 1.2)
    
    w_ei = 0.5
    w_pi = 0.0
    w_ucb = 0.5
    
    combined = w_ei * ei_norm + w_pi * pi_norm + w_ucb * ucb_norm
    
    # Cost penalty
    # cost is in [1, 2].
    # Using 1/cost is linear.
    # Using 1/cost**gamma with gamma > 1 favors cheap points more.
    # But we must be careful not to ignore the interior if the minimizer is there.
    # The minimizer is in the inner 80% [0.1, 0.9]^6.
    # Cost in the center is 1 + 0.5 = 1.5.
    # Cost at corner 0 is 1.0.
    # Ratio 1.5/1.0 = 1.5.
    # If gamma=2, ratio is 2.25.
    # This might be too strong.
    # Let's try gamma = 1.5.
    
    cost_penalty = 1.0 / (cost ** 1.5)
    
    final_scores = combined * cost_penalty
    
    # Ensure finite
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline check
    if _deadline is not None and time.time() > _deadline:
        # Return a simple fallback if we are running out of time
        # Just return EI/cost which is fast
        simple_scores = ei / (cost + eps)
        return np.nan_to_num(simple_scores, nan=0.0, posinf=0.0, neginf=0.0)
        
    return final_scores

# EVOLVE-BLOCK-END
