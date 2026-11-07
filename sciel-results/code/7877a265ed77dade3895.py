"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) acquisition function for Bayesian optimization.
    
    KG estimates the expected reduction in the best observed value after
    evaluating a new candidate. For minimization:
    KG(x) = E[min(f(x), f_best)] - min(mu(x), f_best)
    
    where the expectation is over the Gaussian posterior N(mu(x), sigma(x)^2).
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s limit
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # For KG: KG(x) = E[min(f(x), f_best)] - min(mu(x), f_best)
    # 
    # E[min(f(x), f_best)] where f(x) ~ N(mu, sigma^2):
    # = mu * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    #   + f_best * (1 - Phi((f_best - mu)/sigma))
    #
    # where Phi is the standard normal CDF and phi is the standard normal PDF.
    
    # z-score: how far mu is below f_best in units of sigma
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    # E[min(f(x), f_best)] = mu * Phi(z) + sigma * phi(z) + f_best * (1 - Phi(z))
    E_min = mu * Phi + sigma_safe * phi + f_best * (1 - Phi)
    
    # min(mu(x), f_best)
    min_mu = np.minimum(mu, f_best)
    
    # KG = E[min(f(x), f_best)] - min(mu(x), f_best)
    kg = E_min - min_mu
    
    # Ensure finite scores
    kg = np.where(np.isfinite(kg), kg, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return kg
    
    return kg
# EVOLVE-BLOCK-END
