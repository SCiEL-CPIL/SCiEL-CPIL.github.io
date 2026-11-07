"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    """
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate the improvement term
    # We want to find points where f(x) < f_best
    # Improvement is f_best - f(x)
    improvement = f_best - mu
    
    # Calculate z-score
    z = improvement / sigma
    
    # Calculate EI
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei_scores = improvement * Phi_z + sigma * phi_z
    
    # Handle cases where sigma is very small or improvement is negative
    # If improvement < 0, EI should be 0 (no improvement expected)
    # The formula naturally handles this because Phi(z) becomes small and the term becomes negative?
    # Let's check: if mu > f_best, improvement < 0. z < 0.
    # Phi(z) is small. phi(z) is positive.
    # EI = negative * small + positive * positive.
    # Actually, if mu > f_best, the probability of improvement is low, but the expected improvement is still positive?
    # No, EI is defined as E[max(f_best - f(x), 0)].
    # If f(x) > f_best, max is 0.
    # The formula $(f_{best} - \mu) \Phi(z) + \sigma \phi(z)$ is correct for $z = (f_{best} - \mu)/\sigma$.
    # If $\mu > f_{best}$, $z < 0$.
    # Example: $\mu = 1, f_{best} = 0, \sigma = 1$. $z = -1$.
    # $EI = (-1) \Phi(-1) + 1 \phi(-1) = -0.158 + 0.242 = 0.084 > 0$.
    # This is correct. Even if the mean is worse, there is a chance of improvement.
    
    # Ensure finite scores
    ei_scores = np.where(np.isfinite(ei_scores), ei_scores, 0.0)
    
    return ei_scores
# EVOLVE-BLOCK-END
