import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
        """
        Compute Quantile-based Upper Confidence Bound (UCB) for minimization.
        
        This approach approximates the expected minimum of a large number of
        samples from the posterior distribution by targeting the lower quantile.
        Score(x) = -(mu(x) - z_alpha * sigma(x)) = -mu(x) + z_alpha * sigma(x)
        
        Using a small alpha (1e-4) corresponds to a high z_alpha (~3.72),
        encouraging exploration of regions with high uncertainty.
        """
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        
        # Clip sigma to avoid numerical issues
        sigma = np.maximum(sigma, 1e-12)
        
        # Quantile parameter
        alpha = 1e-4
        z_alpha = norm.ppf(1 - alpha)
        
        # UCB for minimization: maximize -mu + z * sigma
        score = -mu + z_alpha * sigma
        
        # Ensure finite values
        score = np.where(np.isfinite(score), score, -1e18)
        
        return score
