"""Acquisition function for Bayesian optimization of Michalewicz 2D using Expected Improvement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 320  # safety margin before 330s kill

def acquisition(mu, sigma, f_best):
        """
        Expected Improvement (EI) acquisition function for minimization.
        Returns scores for 2000 candidate points. Higher score = more likely to be selected.
        """
        # Deadline guard
        if time.time() - _start_time > _TIME_LIMIT:
            # Pure exploitation fallback
            return -mu
        
        n = len(mu)
        
        # Sanitize inputs
        sigma_safe = np.maximum(sigma, 1e-10)
        
        # Compute Z-score
        # For minimization, improvement is (f_best - f(x))
        # Z = (f_best - mu) / sigma
        z = (f_best - mu) / sigma_safe
        
        # Compute EI
        # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
        phi_z = norm.pdf(z)
        Phi_z = norm.cdf(z)
        
        ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
        
        # Ensure finite scores
        ei[~np.isfinite(ei)] = -1e10
        
        return ei
    # EVOLVE-BLOCK-END
