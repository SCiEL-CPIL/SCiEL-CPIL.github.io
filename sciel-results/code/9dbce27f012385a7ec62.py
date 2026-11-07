"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Probabilistic Expected Improvement with dynamic threshold scaling.
    # Scale the improvement threshold by a factor based on the current best value
    # to balance exploration and exploitation.
    # Use a small epsilon to avoid division by zero.
    eps = 1e-9
    
    # Dynamic threshold scaling: when f_best is close to 0 (minimum), we want
    # more exploration; when f_best is large, we can be more exploitative.
    # The scale factor adjusts the effective threshold.
    # For minimization, EI = E[max(f_best - f(x), 0)]
    # We modify this to use a scaled threshold: f_best - alpha * |f_best| * sigma
    # where alpha controls the balance.
    
    # A simpler approach: use standard EI but with a slight modification
    # to the threshold. The idea is to scale the improvement by the current best.
    # When f_best is near 0, we want to explore more aggressively.
    
    # Let's use a variant where we adjust the threshold dynamically.
    # threshold = f_best * (1 + beta * |f_best|) where beta is small
    # This makes the threshold slightly larger when f_best is large, encouraging exploration.
    
    # Actually, for minimization where the minimum is 0, a better approach:
    # Use EI with a dynamic epsilon that depends on f_best.
    # When f_best is close to 0, use a smaller epsilon (more exploitative).
    # When f_best is far from 0, use a larger epsilon (more explorative).
    
    sigma_safe = np.maximum(sigma, eps)
    
    # Standard EI for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Dynamic scaling: multiply EI by a factor that depends on f_best
    # When f_best is close to 0 (near optimum), we want to be more precise
    # When f_best is large (far from optimum), we want to explore more
    # Scale factor: 1 / (1 + |f_best|) reduces the impact when f_best is large
    # But this might be too aggressive. Let's use a milder scaling.
    
    # Alternative: add a small exploration term that depends on f_best
    # exploration_bonus = gamma * sigma * exp(-|f_best|)
    # This encourages exploration when we're far from the optimum.
    
    # Let's use a combination: standard EI plus a scaled exploration term
    gamma = 0.1  # exploration weight
    exploration = gamma * sigma_safe * np.exp(-np.abs(f_best))
    
    return ei + exploration
# EVOLVE-BLOCK-END
