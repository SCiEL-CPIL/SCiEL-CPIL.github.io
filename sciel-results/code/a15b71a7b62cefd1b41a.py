import numpy as np

def acquisition(mu, sigma, f_best):
        """
        Compute the Gaussian Process Lower Confidence Bound (GP-LCB) acquisition function.
        
        Parameters:
        mu : np.ndarray
            Posterior mean at candidate points, shape (N,).
        sigma : np.ndarray
            Posterior standard deviation at candidate points, shape (N,).
        f_best : float
            Best observed value so far (minimum).
        
        Returns:
        scores : np.ndarray
            LCB scores, shape (N,), higher is better.
        """
        # Handle numerical stability for sigma
        sigma_safe = np.maximum(sigma, 1e-9)
        
        # GP-LCB: score = -mu - kappa * sigma
        # Maximizing this is equivalent to minimizing mu + kappa * sigma
        # We want to find low values of the objective, so we look for low mu.
        # The -mu term encourages exploitation of low mean values.
        # The -sigma term encourages exploration of high uncertainty regions.
        kappa = 2.0
        scores = -mu - kappa * sigma_safe
        
        # Ensure finite values
        scores = np.where(np.isfinite(scores), scores, 0.0)
        
        return scores
