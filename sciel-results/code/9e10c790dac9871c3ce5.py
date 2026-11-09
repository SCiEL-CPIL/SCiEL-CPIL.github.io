"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're running out of time, fall back to simple greedy
    if time.time() - _START_TIME > _TIME_LIMIT:
        mu_arr = np.asarray(mu, dtype=np.float64)
        cost_arr = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost_arr, 1e-6)
        return -mu_arr / cost_safe

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs
    cost = np.maximum(cost, 1e-6)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # f_best is the lowest observed value
    fb = float(f_best)
    
    # Adaptive strategy:
    # 1. Compute Expected Improvement (EI) for minimization
    # 2. Compute UCB-style exploration term
    # 3. Blend based on how far we are from the optimum
    #
    # Rastrigin 2D global min is 0. Typical local minima are around 8-10.
    # If f_best is high, we need more exploration. If f_best is low, exploit.
    
    # EI for minimization:
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    
    z = (fb - mu) / sigma_safe
    # Clip z to avoid numerical issues in extreme tails
    z_clipped = np.clip(z, -10, 10)
    
    ei = (fb - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    # EI should be non-negative; clip small negatives from numerical error
    ei = np.maximum(ei, 0.0)
    
    # UCB exploration term for minimization:
    # We want to explore where sigma is high.
    # The "optimistic" value is mu - beta * sigma.
    # The gain from optimism relative to f_best is: f_best - (mu - beta*sigma) = f_best - mu + beta*sigma
    # But we want a normalized exploration score.
    
    # Adaptive beta:
    # When f_best is large (e.g., > 5), use higher beta for exploration.
    # When f_best is small (e.g., < 1), use lower beta for exploitation.
    # Rastrigin values: global min 0, local mins ~8-10.
    # Use a smooth transition.
    
    # Normalize f_best: if fb <= 0, fully exploit. If fb >= 10, fully explore.
    t = np.clip(fb / 10.0, 0.0, 1.0)
    beta_min = 0.5
    beta_max = 2.5
    beta = beta_min + (beta_max - beta_min) * t
    
    # UCB gain: f_best - mu + beta * sigma
    ucb_gain = fb - mu + beta * sigma_safe
    
    # Combine EI and UCB gain.
    # Both are "gains" (higher is better).
    # We want to weight exploration more when uncertain and exploitation when confident.
    # Use a blend: score = w * EI + (1-w) * ucb_gain
    # w should be high when we're close to optimum (exploit), low when far (explore).
    # Actually, EI already adapts: when sigma is small, EI ≈ max(f_best - mu, 0).
    # When sigma is large, EI is more conservative than UCB.
    
    # Let's use a hybrid: score = EI + alpha * sigma, where alpha is small
    # This adds a small exploration bonus on top of EI.
    # Or better: use the maximum of EI and a scaled UCB term.
    
    # Alternative: Use a weighted combination that shifts from UCB to EI as we improve.
    # When fb is high, ucb_gain dominates. When fb is low, EI dominates (since ucb_gain ≈ fb - mu when sigma→0).
    
    # Let's try: score = EI + gamma * sigma, where gamma adapts
    # This is like EI with an extra exploration term.
    
    # Actually, the most principled approach:
    # Use EI as the base, but add a small UCB exploration term weighted by uncertainty.
    # score = EI + delta * (sigma / (sigma + epsilon)) * (fb - mu + beta*sigma)
    # This is getting complex. Let's keep it simpler.
    
    # Simple effective combination:
    # score = EI * w_ei + ucb_gain * w_ucb
    # where w_ei and w_ucb depend on fb.
    
    # When fb is large, we don't trust the GP much, so UCB is better.
    # When fb is small, EI is more reliable.
    
    # Linear blend based on t = fb/10
    w_ucb = 0.5 * t  # 0 to 0.5
    w_ei = 1.0 - w_ucb  # 1.0 to 0.5
    
    # But we should normalize the scales. EI and ucb_gain have similar scales.
    # Let's just do:
    raw_score = w_ei * ei + w_ucb * ucb_gain
    
    # Cost-aware: divide by cost
    scores = raw_score / cost
    
    # Ensure finite outputs
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
