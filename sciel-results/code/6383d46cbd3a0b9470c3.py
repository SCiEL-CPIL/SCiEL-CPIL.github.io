# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    # Time-adaptive hybrid EI: boost exploration early, then exploit.
    # We don't know the exact step count, but we can infer from sigma magnitude
    # and f_best. Early: sigma is large relative to f_best, exploration is needed.
    # Late: sigma is small, exploitation is better.
    
    # Use a simple heuristic: if f_best is still large (> 10), we're early.
    # The Beale function minimum is 0, and values at random points are typically > 100.
    # As we find good points, f_best drops.
    
    # EI computation
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small UCB-like exploration term that decays as f_best improves
    # When f_best is large, we add more exploration
    # When f_best is small, we rely more on EI (exploitation)
    
    # Exploration weight based on how far f_best is from the optimum
    # f_best starts around 100-200 for Beale, ends near 0
    # Use a smooth decay: exp(-f_best / scale)
    # When f_best = 100, weight is tiny (exploit)
    # When f_best = 10, weight is moderate
    # When f_best = 1, weight is high (explore)
    # Actually, we want the opposite: early on (f_best large), explore more
    # Let's use: weight = min(1.0, f_best / 50.0)
    # When f_best = 100, weight = 1.0 (full exploration boost)
    # When f_best = 50, weight = 1.0
    # When f_best = 10, weight = 0.2
    # When f_best = 1, weight = 0.02
    
    # UCB term: mu - beta * sigma (we want high score, so high EI is good)
    # Actually, for minimization, UCB would be: mu - beta * sigma
    # But we're maximizing the acquisition, so we want to encourage exploration
    # Let's add: beta * sigma as an exploration bonus
    # This encourages evaluating points with high uncertainty
    
    # Adaptive beta: larger when f_best is large (early), smaller when f_best is small
    beta = 0.5 * np.exp(-f_best / 20.0)
    
    # Combined score: EI + beta * sigma
    # This adds exploration pressure early on
    scores = ei + beta * sigma
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
