# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Compute a hybrid acquisition score combining Expected Improvement (EI)
    and a quantile-based exploration term for minimization.
    
    The strategy combines:
    1. EI (from inspiration 2/3): Exploits the GP posterior by targeting
       points with high probability of beating f_best.
    2. Quantile (from directive): Uses a very small alpha (1e-4) to target
       the lower tail of the posterior, approximating the global lower bound.
    
    Score = EI + gamma * (f_best - Q_alpha(x))
    where Q_alpha(x) = mu(x) + z_alpha * sigma(x) and z_alpha = norm.ppf(alpha).
    
    This hybrid balances local exploitation (EI) with global exploration
    towards the theoretical minimum (quantile), which is particularly
    effective for the 1D sphere function where the global minimum is known
    to be 0.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Clip sigma to avoid division by zero and numerical instability
    sigma = np.maximum(sigma, 1e-12)
    
    fb = float(f_best)
    
    # Compute Expected Improvement for minimization
    # EI(x) = (fb - mu) * Phi(z) + sigma * phi(z)
    # where z = (fb - mu) / sigma
    z = (fb - mu) / sigma
    cdf_vals = norm.cdf(z)
    pdf_vals = norm.pdf(z)
    ei = (fb - mu) * cdf_vals + sigma * pdf_vals
    
    # Compute quantile-based exploration term
    # Use a very small alpha to target the lower tail
    alpha = 1e-4
    z_alpha = norm.ppf(alpha)  # Approximately -3.72
    
    # Q_alpha(x) = mu + z_alpha * sigma
    # Since z_alpha is negative, this is the lower quantile
    quantile = mu + z_alpha * sigma
    
    # The improvement potential from the quantile perspective
    # We want to maximize (fb - quantile), which is the expected reduction
    # if we could achieve the alpha-quantile
    quantile_improvement = fb - quantile
    
    # Scale the quantile term to be comparable to EI
    # EI is typically on the order of sigma * phi(z), which is small
    # The quantile improvement can be larger, so we scale it down
    # Use a small gamma to let EI dominate but add a global exploration bias
    gamma = 0.1
    
    score = ei + gamma * quantile_improvement
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -1e18)
    
    return score
# EVOLVE-BLOCK-END
