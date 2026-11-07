"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT_LIMIT = 290.0  # Safety margin before 300s

def acquisition(mu, sigma, f_best, cost):
    # Time guard: if we are close to the deadline, return a simple score
    # to ensure quick termination of the current step or to prioritize exploitation.
    if time.time() - _start_time > _TIMEOUT_LIMIT:
        return -mu / cost

    # Ensure sigma is strictly positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Cost-aware acquisition using a hybrid of Expected Improvement (EI)
    # and Probability of Improvement (PI), with a cost penalty.
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Probability of Improvement (scaled)
    pi = norm.cdf(z)
    
    # Hybrid: blend EI and PI. EI gives magnitude of improvement,
    # PI gives confidence. A weighted sum helps balance exploration/exploitation.
    # Also, we can use a "sharpness" factor that emphasizes points with
    # high improvement relative to uncertainty.
    
    # Alternative: Use the "Upper Confidence Bound" (UCB) form but adapted
    # for minimization and cost-awareness.
    # UCB for minimization: mu - kappa * sigma
    # But we want cost-awareness: (f_best - mu + kappa * sigma) / cost
    
    # Let's try a more sophisticated approach:
    # 1. Compute the "gain" as f_best - mu (positive when mu < f_best)
    # 2. Compute the "confidence" as 1 / (1 + sigma) or similar
    # 3. Combine: gain * confidence / cost
    
    # Actually, let's stick with a well-tuned cost-aware EI but add
    # an exploration bonus that decays with sigma.
    
    # The key insight from inspiration 1 (score 1.2643) is cost normalization.
    # Let's enhance the cost-awareness by using a power of cost in the denominator.
    
    # Score = EI / (cost ** alpha) where alpha > 1 penalizes expensive points more
    # Or: Score = EI * (1 / cost) * (1 + beta * sigma / f_best) to boost exploration
    
    # Let's try: score = ei / cost, but also add a term that favors
    # points with lower sigma (more certain) when they are near the best.
    
    # A robust approach: use the "Expected Hypervolume Improvement" (EHVI)
    # approximation, but for single-objective, this reduces to EI-like forms.
    
    # Let's use a modified EI that accounts for the fact that we're minimizing:
    # The standard EI is correct for minimization when f_best is the minimum observed.
    
    # To improve over the seed, let's add a small exploration bonus
    # that is proportional to sigma but damped to avoid pure random search.
    exploration_bonus = 0.1 * sigma
    
    score = (ei + exploration_bonus) / cost
    
    # Handle any potential NaNs or Infs
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
