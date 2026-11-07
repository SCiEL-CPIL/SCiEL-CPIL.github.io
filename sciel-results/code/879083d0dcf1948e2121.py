"""Acquisition function for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for minimizing sphere 1d on [-5, 5].
    Uses a combination of Expected Improvement and UCB with adaptive weighting.
    """
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB (for minimization, we subtract kappa*sigma)
    # Adaptive kappa: larger when we have high uncertainty
    kappa = 1.5
    ucb = mu - kappa * sigma_safe
    
    # Probability of Improvement (scaled)
    pi = norm.cdf(z)
    
    # Combine EI and UCB. The relative scaling matters.
    # EI is in objective units, UCB is also in objective units but with a different scale.
    # Let's normalize EI to be comparable to UCB.
    # Alternatively, just use a weighted sum with carefully chosen weights.
    
    # A common approach is to use EI with an exploration parameter, or to
    # blend EI and UCB. Since both are in the same units (objective value),
    # we can combine them directly.
    
    # Let's try: score = alpha * EI + beta * (-UCB)
    # Where -UCB is higher when we want to explore (lower mu + higher sigma)
    # Actually, UCB for minimization: we want to minimize mu - kappa*sigma
    # So higher values of (mu - kappa*sigma) are worse. We want to MAXIMIZE the acquisition.
    # So acquisition = -UCB = -mu + kappa*sigma
    
    # Let's use a blend:
    # score = EI + weight * (f_best - mu + kappa * sigma)
    # But EI already captures the improvement over f_best.
    
    # Simpler: use EI with a small exploration bonus
    # Or use UCB directly if EI is not working well.
    
    # Let's try a hybrid: primarily EI, with a small UCB component
    # The key is that EI tends to be greedy, UCB is more exploratory.
    
    # Weight UCB relative to EI's typical scale
    # EI is roughly on the scale of sigma when z is moderate
    # UCB is on the scale of mu + sigma
    
    # Let's normalize both to have similar magnitudes
    # Actually, let's just use EI with an exploration parameter xi
    # EI with xi: z = (f_best - xi - mu) / sigma
    # This shifts the improvement threshold
    
    # Try xi = 0.1 * (f_best - min(mu)) to adapt exploration
    mu_min = np.min(mu)
    mu_range = np.max(mu) - mu_min + 1e-8
    xi = 0.05 * mu_range  # Small exploration parameter
    
    z_ei = (f_best - xi - mu) / sigma_safe
    ei_xi = (f_best - xi - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # Also compute a UCB-like score
    # For minimization, we want to go where mu is low and sigma is high
    # ucb_score = -mu + kappa * sigma (we want to maximize this)
    kappa = 1.0
    ucb_score = -mu + kappa * sigma_safe
    
    # Combine: use EI as primary, UCB as secondary
    # Scale them to be comparable
    ei_scale = np.std(ei_xi) + 1e-8
    ucb_scale = np.std(ucb_score) + 1e-8
    
    # Normalize
    ei_norm = ei_xi / ei_scale
    ucb_norm = ucb_score / ucb_scale
    
    # Weight: 80% EI, 20% UCB
    score = 0.8 * ei_norm + 0.2 * ucb_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
