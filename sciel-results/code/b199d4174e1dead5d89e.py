"""
Bayesian Optimization Acquisition using Thompson Sampling.

The task is to minimize a fixed seeded random RBF mixture in 2D.
We use a GP posterior (mu, sigma) provided by the evaluator.
Thompson Sampling is chosen because it is often superior to EI/UCB for multimodal functions,
as it samples from the posterior distribution of the objective, allowing it to
navigate multiple basins of attraction effectively.

We "optimize" the exploration-exploitation trade-off by using the natural
properties of the GP posterior (sigma). If we want to be more aggressive, we could scale sigma,
but pure TS is a strong baseline.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global time tracking for deadline guard
_T0 = time.time()
_DEADLINE = 325.0  # 330s limit, 5s margin
_EPS = 1e-12

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    
    EI is a deterministic acquisition function that computes the expected
    improvement over the current best. It balances exploitation (low mu)
    with exploration (high sigma) in a principled way.
    
    For minimization: EI(x) = (f_best - mu(x)) * Phi(d) + sigma(x) * phi(d)
    where d = (f_best - mu(x)) / sigma(x)
    
    Args:
        mu: (2000,) posterior mean
        sigma: (2000,) posterior std
        f_best: float, lowest observed value
    
    Returns:
        (2000,) scores. Higher is better.
    """
    # Deadline check
    if time.time() - _T0 > _DEADLINE:
        # Fallback to exploitation-focused score
        return -(mu - f_best)

    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, _EPS)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, _EPS)
    mu_safe = np.where(np.isfinite(mu), mu, f_best)

    # Compute d = (f_best - mu) / sigma
    d = (f_best - mu_safe) / sigma_safe
    
    # EI = (f_best - mu) * Phi(d) + sigma * phi(d)
    # Phi(d) = norm.cdf(d)
    # phi(d) = norm.pdf(d)
    phi_d = norm.pdf(d)
    Phi_d = norm.cdf(d)
    
    ei = (f_best - mu_safe) * Phi_d + sigma_safe * phi_d
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
