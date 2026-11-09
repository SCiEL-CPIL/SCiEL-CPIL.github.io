"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Handle potential numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Adaptive exploration weight
    # The Beale function has a minimum of 0 and values can range up to ~50-100 in the domain.
    # Early on, f_best is high (e.g., 20-50), we want to explore.
    # Late on, f_best is low (e.g., <1), we want to exploit.
    
    # We combine the structural idea of the current program (adaptive beta scaling)
    # with the refinement machinery of the inspirations (EI + exploration bonus).
    
    # Let's use a hybrid approach:
    # 1. Use EI as the base score.
    # 2. Add a small UCB-like exploration term: beta * sigma.
    # 3. The beta should decay as f_best decreases.
    
    # A good balance is to use a beta that is large early and small late.
    # beta = 1.0 + 1.0 * (f_best / (f_best + 1.0))
    # If f_best = 100, beta ~ 2.0
    # If f_best = 1, beta ~ 1.5
    # If f_best = 0.1, beta ~ 1.09
    
    # Alternatively, use the sqrt scaling from the current program but with a smaller base.
    # beta = 0.5 * np.sqrt(f_best + 0.1) + 0.1
    # If f_best = 50, beta = 0.5 * 7.07 + 0.1 = 3.635
    # If f_best = 1, beta = 0.5 * 1.05 + 0.1 = 0.625
    # If f_best = 0.1, beta = 0.5 * 0.45 + 0.1 = 0.325
    
    # Let's try the sigmoid-like scaling from Inspiration 1, as it provides a smooth transition.
    # But we can also combine it with a small constant to ensure some exploration even when f_best is very small.
    # beta = 0.1 + 0.4 * (f_best / (f_best + 1.0))
    
    # Let's use a slightly more aggressive exploration in the early stages.
    # beta = 0.2 + 0.8 * (f_best / (f_best + 1.0))
    
    # However, the current program's sqrt scaling might be more robust for a wider range of f_best values.
    # Let's use a combination: start with the sqrt scaling but cap it to avoid excessive exploration.
    # beta = np.minimum(1.5 * np.sqrt(f_best + 0.1), 5.0)
    
    # Let's go with the sigmoid-like scaling as it's smoother and well-behaved.
    # beta = 0.1 + 0.4 * (f_best / (f_best + 1.0))
    
    # To make it more adaptive, let's use a slightly larger coefficient.
    # beta = 0.2 + 0.6 * (f_best / (f_best + 1.0))
    
    # Final choice: Use a moderate exploration bonus that decays smoothly.
    beta = 0.2 + 0.6 * (f_best / (f_best + 1.0))
    
    # Score = EI + beta * sigma
    # This adds an explicit exploration bonus on top of EI.
    # EI already rewards high sigma when mu is high, but this strengthens it.
    
    score = ei + beta * sigma
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
