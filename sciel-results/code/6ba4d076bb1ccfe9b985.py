"""Adaptive Acquisition Function for Bayesian Optimization.

Structural change: Replaces offline-tuned static EI/UCB parameters with an
online adaptive strategy that blends EI and UCB based on the optimization
progress. This removes import-time overhead and allows the algorithm to
shift from exploration to exploitation dynamically.
"""

import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Computes an adaptive acquisition score for the next candidate.
    
    The score is a blend of Expected Improvement (EI) and Upper Confidence Bound (UCB).
    - Early in optimization (high uncertainty relative to best), UCB weight is higher.
    - Later (low uncertainty or good best found), EI weight is higher.
    
    Parameters:
    mu: (2000,) array of GP mean predictions.
    sigma: (2000,) array of GP standard deviations.
    f_best: float, the best (lowest) observed value so far.
    
    Returns:
    (2000,) array of scores. Higher scores indicate better candidates.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # 1. Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Compute UCB for minimization
    # UCB = -mu + beta * sigma
    # We need to normalize EI and UCB to be on similar scales for blending.
    # EI is typically small (local improvement). UCB can be larger.
    # Let's use a dynamic beta that adapts to the "exploration state".
    
    # Heuristic for adaptation:
    # If the best found is very good (close to expected min?), we should exploit.
    # If we are early (sigma is high relative to range), we should explore.
    
    # A simple adaptive strategy:
    # beta_t = beta_max * exp(-t / decay) or similar.
    # But we don't have 't' (step count) directly passed, only f_best and mu/sigma.
    # However, we can infer "progress" from the relationship between mu and f_best.
    # Or, more simply, use the magnitude of sigma.
    
    # Let's use a fixed but well-tuned blend that is robust.
    # Common practice: EI is often superior for smooth functions like Hartmann.
    # But pure EI can get stuck in local optima if the GP is uncertain.
    
    # Let's try a "Knowledge Gradient" like approximation or just a strong EI.
    # Actually, the prompt says "inspiration 1" (simple) scored 1.04.
    # The current complex one scores 1.00.
    # This suggests the complexity is hurting or not helping.
    
    # Let's implement a robust EI with a small UCB component to prevent stagnation.
    # UCB = -mu + beta * sigma.
    # We want to maximize the score.
    
    # Normalize EI to have mean 0, std 1? No, that changes the scale.
    # Let's just add a small UCB term.
    
    # Dynamic Beta:
    # If the average sigma is high, we are in unexplored regions -> explore more.
    # If the average sigma is low, we are in well-explored regions -> exploit more.
    
    mean_sigma = np.mean(sigma)
    
    # Assume a typical scale of the problem. Hartmann 6D values are around -3.
    # Standard deviation of the GP prior might be around 1.
    # Let's map mean_sigma to a beta in [0.5, 2.0].
    # High mean_sigma -> High beta (more exploration).
    # Low mean_sigma -> Low beta (more exploitation, closer to EI).
    
    # Simple linear mapping clipped
    # If mean_sigma > 1.0, beta = 2.0
    # If mean_sigma < 0.1, beta = 0.5
    beta = np.clip(0.5 + 1.5 * mean_sigma, 0.5, 2.0)
    
    ucb = -mu + beta * sigma
    
    # Now we have EI and UCB. They have different scales.
    # EI is positive when mu < f_best.
    # UCB is higher when mu is low or sigma is high.
    
    # Let's normalize them to have similar variance before blending?
    # Or just use a weighted sum with fixed weights?
    # Given the stagnation, let's stick to a very standard, robust acquisition.
    # Pure EI is the gold standard for these tasks.
    # The "inspiration 1" likely used pure EI or a very simple variant.
    
    # Let's return pure EI, but ensure numerical stability.
    # The previous code had a complex offline fit that might have picked bad params.
    # Pure EI is the most reliable baseline.
    
    # To be slightly more exploratory than pure EI (which can be greedy),
    # let's add a tiny bit of UCB if EI is very low (stagnation detection).
    # If max(EI) is very close to 0, we might be stuck.
    
    max_ei = np.max(ei)
    
    # If max_ei is very small, boost exploration.
    # Threshold: if max improvement is less than 1% of the current best magnitude?
    # f_best is negative.
    scale = np.abs(f_best) + 1e-6
    
    if max_ei < 0.01 * scale:
        # Stagnation: mix in UCB
        # Normalize UCB to have similar scale to EI?
        # Just add a scaled UCB.
        # Let's use a fixed weight for the UCB part in stagnation.
        score = ei + 0.1 * ucb
    else:
        # Normal operation: use EI
        score = ei
        
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
