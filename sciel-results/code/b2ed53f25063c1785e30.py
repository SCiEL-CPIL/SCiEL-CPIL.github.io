import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    f_best = float(f_best)

    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    # Use stable computation for norm.cdf and norm.pdf
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    EI = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Calculate UCB for minimization
    # We want to minimize, so we prefer low mu and high sigma.
    # Score_ucb = -mu + beta * sigma
    # We need to scale EI and UCB to be comparable.
    # EI is in units of f. UCB is in units of f.
    # So we can just add them.
    
    # Adaptive beta: larger beta when f_best is large (early), smaller when f_best is small (late)
    # This helps with convergence.
    # If f_best is 25, we are far from 0. Explore more.
    # If f_best is 0.1, we are close. Exploit more.
    # Let's use a simple scaling.
    # beta = 2.0 + 2.0 * (f_best / 25.0)  # Rough heuristic
    
    # Alternatively, just use a fixed beta that works well for sphere.
    # For 1D sphere, a moderate beta works.
    beta = 2.5
    
    UCB = -mu + beta * sigma
    
    # Combine
    # EI is usually the dominant term for convergence.
    # UCB helps escape local minima if GP is wrong.
    # Let's weight them.
    score = EI + 0.5 * UCB
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -1e12)
    
    return score
