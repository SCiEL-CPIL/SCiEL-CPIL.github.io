import numpy as np
import time
from scipy.stats import norm

_T0 = time.time()
_DEADLINE = _T0 + 325.0  # Safety margin vs 330s kill

# Precompute the z-score for the specified quantile alpha = 1e-4
# This corresponds to the lower tail of the standard normal distribution.
# We use the negative of this z-score as the exploration weight (beta)
# for the acquisition function score: -mu + beta * sigma
_ALPHA = 1e-4
_Z_ALPHA = norm.ppf(_ALPHA)
_BETA = -_Z_ALPHA  # Positive value, approx 3.719

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a fixed posterior quantile.
    
    This approach models the acquisition as targeting the lower quantile
    of the posterior distribution, approximating the expected minimum
    of a large number of samples from the posterior.
    
    Score(x) = -Q_alpha(x) = -(mu(x) + z_alpha * sigma(x))
             = -mu(x) - z_alpha * sigma(x)
             = -mu(x) + beta * sigma(x)
    
    where beta = -z_alpha.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback to exploitation only if time is running out
        mu = np.asarray(mu, dtype=float)
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        return scores

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is finite and non-negative
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate scores
    # Higher score means more promising for minimization
    scores = -mu + _BETA * sigma
    
    # Handle non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
