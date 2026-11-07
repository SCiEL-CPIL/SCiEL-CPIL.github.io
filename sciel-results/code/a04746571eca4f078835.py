"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining Expected Improvement (EI) and
    Upper Confidence Bound (UCB) to balance exploitation and exploration.
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound (beta = 1.0)
    ucb = mu + 1.0 * sigma_safe
    
    # Combine EI and UCB. 
    # Note: EI is non-negative (higher is better for minimization via (f_best - mu) logic).
    # UCB is lower when mu is low or sigma is high. 
    # We want to maximize the score.
    # EI encourages points where f_best - mu is large (mu is low) and sigma is moderate.
    # UCB encourages points where mu is low and sigma is high.
    # Let's weight them.
    # To make them comparable, we can just sum them or use a weighted sum.
    # Since EI is in objective units and UCB is in objective units, summing is reasonable.
    # However, EI can be very small if mu is close to f_best, while UCB can be large.
    # A common hybrid is to take the max or a weighted sum.
    # Let's try a weighted sum: score = 0.5 * ei + 0.5 * (-ucb) ? No, UCB is for minimization usually as mu + beta*sigma, so lower is better.
    # Wait, the task says "higher meaning evaluate next".
    # So we need to convert UCB to a "higher is better" score.
    # For minimization, UCB value is mu + beta*sigma. We want to minimize this.
    # So a score based on UCB would be - (mu + beta*sigma).
    # But EI is already "higher is better" for minimization (it measures expected improvement over f_best).
    
    # Let's refine:
    # EI score: (f_best - mu) * Phi(z) + sigma * phi(z). This is >= 0.
    # UCB score: - (mu + beta * sigma). This can be negative or positive.
    # If we just add them, the scales might not match well.
    # Let's try: score = ei - (mu + beta * sigma)
    # Or perhaps normalize? No, keep it simple first.
    
    # Actually, a very effective strategy for multi-modal functions like Goldstein-Price is to ensure sufficient exploration.
    # Let's use a weighted combination where exploration is slightly favored initially or balanced.
    # Let's try: score = ei + 0.5 * (-mu - 1.0 * sigma)
    # This rewards low mu (via -mu) and high sigma (via -sigma).
    
    # Alternative: Use the maximum of EI and UCB-based score.
    # ucb_score = -(mu + 1.0 * sigma)
    # return np.maximum(ei, ucb_score)
    
    # Let's go with a weighted sum for smoothness, but tuned for exploration.
    # score = ei - 1.0 * mu - 0.5 * sigma
    # This effectively creates a lower bound on the function value that accounts for uncertainty.
    
    # Let's try a simpler, robust hybrid:
    # score = ei + 0.5 * sigma - mu
    # This is roughly: (f_best - mu)*... + 0.5*sigma - mu.
    # The -mu term strongly drives toward low mean.
    # The +0.5*sigma term encourages exploration.
    # The EI term adds a bonus for expected improvement.
    
    score = ei + 0.5 * sigma_safe - mu
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
