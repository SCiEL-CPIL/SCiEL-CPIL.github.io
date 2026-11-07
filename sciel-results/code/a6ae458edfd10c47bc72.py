import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Hybrid acquisition: blend Expected Improvement with a UCB exploration bonus.
    # EI drives exploitation toward points that are likely to beat f_best.
    # The UCB term (beta * sigma) encourages exploration of uncertain regions.
    # Using a smooth combination keeps the score finite and well-behaved.
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Scale EI so it's comparable to the UCB term.
    # When f_best is large (early optimization), EI can be large.
    # When f_best is small (near optimum), EI is small and UCB dominates.
    # We use a fixed beta that works reasonably across the optimization.
    beta = 1.5
    ucb_term = beta * sigma_safe
    
    # Combine: primarily EI, with a moderate exploration bonus.
    # This helps escape flat regions where EI ≈ 0 but sigma is still significant.
    score = ei + 0.1 * ucb_term
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
