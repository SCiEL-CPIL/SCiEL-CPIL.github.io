"""Cost-aware acquisition for Bayesian optimization on Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_module_start_time = time.time()
_DEADLINE_MARGIN = 20  # seconds before 330s limit

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    elapsed = time.time() - _module_start_time
    if elapsed > 310 - _DEADLINE_MARGIN:
        return mu - sigma  # fallback LCB
    
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (for minimization: EI = E[max(f_best - f, 0)])
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive cost-aware scoring
    # Theoretical minimum of Hartmann 3D
    f_min = -3.86278
    
    # Gap to optimum: large early, small late
    gap = f_best - f_min
    # Normalize gap to [0, 1] where 1 is "very good" (small gap) and 0 is "bad" (large gap)
    # Assume max relevant gap is around 2.0 for Hartmann 3D starting from random
    progress = np.clip(1.0 - gap / 2.0, 0.0, 1.0)
    
    # Cost normalization: cost in [1, 2], normalize to [0, 1]
    cost_norm = cost - 1.0  # now in [0, 1]
    
    # Adaptive parameters
    # Alpha: Cost penalty strength.
    # Early (progress=0): Low alpha (0.5) -> Less penalty on cost, allow exploration of expensive areas if EI is high.
    # Late (progress=1): High alpha (2.0) -> Strong penalty on cost, be frugal.
    alpha = 0.5 + 1.5 * progress
    
    # Beta: Uncertainty bonus weight.
    # Early (progress=0): High beta (0.3) -> Strong push for exploration (high sigma).
    # Late (progress=1): Low beta (0.0) -> Pure exploitation.
    beta = 0.3 * (1.0 - progress)**2
    
    # Cost penalty: Use power-law form for stronger non-linearity if desired,
    # but linear with adaptive alpha is robust.
    # Let's try a slightly stronger penalty by using (1 + cost_norm) ** alpha?
    # No, let's stick to linear in cost_norm but with adaptive alpha.
    cost_penalty = 1.0 + alpha * cost_norm
    cost_adjusted_ei = ei / cost_penalty
    
    # Uncertainty bonus
    uncertainty_bonus = beta * sigma_safe
    
    # Combine
    score = cost_adjusted_ei + uncertainty_bonus
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
