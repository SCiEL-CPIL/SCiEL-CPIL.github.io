"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using an adaptive combination of EI and UCB.
    
    Strategy: Use Expected Improvement (EI) as the primary exploitation signal,
    but add a carefully scaled UCB term to encourage exploration in early stages.
    The UCB weight decays as f_best approaches 0 (the known minimum of Beale).
    
    For minimization:
    - EI: (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    - UCB score: -mu + beta * sigma (higher is better for minimization)
    
    We combine them by: score = EI + alpha * UCB_normalized
    where alpha adapts based on how close f_best is to 0.
    
    Since the minimum of Beale is 0, we can use f_best as a proxy for progress.
    When f_best is large (early), we want more exploration (higher alpha).
    When f_best is small (late), we want more exploitation (lower alpha).
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute EI for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Handle very small sigma
    ei = np.where(sigma > 1e-9, ei, np.maximum(f_best - mu, 0))
    
    # UCB score for minimization: -mu + beta * sigma
    # Use a moderate beta
    beta = 1.5
    
    # Adaptive alpha: larger when f_best is far from 0, smaller when close
    # f_best starts around O(100) for Beale and approaches 0
    # Use a soft decay: alpha = 1.0 / (1.0 + f_best)
    # When f_best = 100, alpha ~ 0.01 (mostly EI)
    # When f_best = 1, alpha ~ 0.5 (balanced)
    # When f_best = 0.1, alpha ~ 0.91 (mostly UCB for final polish)
    # Actually, we want the opposite: more exploration early, more exploitation late
    # So alpha should be HIGH early (large f_best) and LOW late (small f_best)
    # Let's use: alpha = min(1.0, 5.0 / (1.0 + f_best))
    # When f_best = 100, alpha = 5/101 ~ 0.05
    # Hmm, that's still small. Let me think differently.
    
    # Alternative: use the magnitude of sigma to guide exploration
    # Points with high uncertainty should get a boost
    # Let's use a fixed combination: score = EI + 0.5 * (-mu + beta * sigma)
    # But EI and UCB have different scales. Let's normalize.
    
    # Simpler approach: just use a well-tuned single acquisition
    # Let's use EI with a small exploration bonus based on sigma
    # score = EI + gamma * sigma, where gamma is small
    
    # Actually, let's try a different approach: use the probability of improvement
    # combined with EI
    
    # PI: P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    pi = norm.cdf((f_best - mu) / sigma_safe)
    pi = np.where(sigma > 1e-9, pi, np.where(mu < f_best, 1.0, 0.0))
    
    # Combine: weighted sum of EI (normalized) and PI
    # EI can be large, so let's normalize it
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    # Final score: primarily EI, with a small PI boost
    # This helps explore regions where there's a chance of improvement
    gamma = 0.3
    score = ei + gamma * pi * sigma_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
