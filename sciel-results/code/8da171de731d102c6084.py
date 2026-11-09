"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected improvement with adaptive exploration weight.
    # Use a small epsilon to avoid division by zero.
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive exploration: when f_best is close to 0 (good), we want more exploitation.
    # When f_best is far from 0, we want more exploration.
    # A simple heuristic: scale EI by a factor that depends on f_best.
    # Since f_best is the minimum observed value, and the global minimum is 0,
    # f_best should be >= 0.
    # If f_best is large, we are far from optimum, so exploration is good.
    # If f_best is small, we are close, so exploitation is better.
    
    # Let's try a weight that decreases as f_best approaches 0.
    # weight = 1 / (1 + f_best)
    # This makes the acquisition function behave like EI when f_best is large,
    # and like 0 when f_best is 0? No, that's not right.
    
    # Actually, EI already handles this. If f_best is close to mu, EI is small.
    # Maybe the issue is that EI is too greedy.
    
    # Let's try a mix of EI and UCB.
    # UCB = mu + beta * sigma
    # We want to minimize, so UCB for minimization is mu - beta * sigma? No, that's for maximizing.
    # For minimization, we want low mu and high sigma.
    # So a good score is -mu + beta * sigma.
    
    # Let's try: score = EI + gamma * sigma
    # This encourages exploration.
    
    # Or maybe just use EI but with a different normalization.
    
    # Let's stick with EI but ensure numerical stability.
    return ei
# EVOLVE-BLOCK-END
