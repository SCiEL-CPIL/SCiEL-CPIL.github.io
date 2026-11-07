import numpy as np

def acquisition(mu, sigma, f_best):
        """
        Acquisition function using Lower Confidence Bound (LCB) for minimization.
        Uses a fixed exploration constant k=3.0.
        """
        sigma = np.maximum(sigma, 1e-9)
        k = 3.0
        scores = -mu + k * sigma
        scores = np.where(np.isfinite(scores), scores, -1e10)
        return scores
