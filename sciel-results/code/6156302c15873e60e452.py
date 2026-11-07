"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Beale 2D.
    
    Uses a pure adaptive Lower Confidence Bound (LCB) where the exploration
    parameter beta scales smoothly with |f_best|. When f_best is large
    (far from the global minimum at 0), beta is large, strongly favoring
    high-uncertainty regions to escape local basins. As f_best approaches 0,
    beta decreases toward a small floor, shifting to exploitation.
    
    The score is -LCB = -mu + beta*sigma, so higher scores are selected.
    """
    start_time = time.time()
    deadline = start_time + 280.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-9)

    if not np.isfinite(f_best):
        f_best = 1e6

    # Adaptive beta: smooth saturation function of |f_best|
    # beta = beta_min + (beta_max - beta_min) * (1 - exp(-|f_best| / tau))
    # This gives beta ~ beta_min when f_best is near 0, and beta ~ beta_max
    # when |f_best| >> tau.
    #
    # For Beale 2D on [-4.5, 4.5]^2, the function values range roughly
    # from 0 to ~30. We want:
    #   - f_best ~ 0:  beta ~ 0.5 (exploit)
    #   - f_best ~ 5:  beta ~ 3.0 (balanced)
    #   - f_best > 15: beta ~ 5.0 (explore)
    beta_min = 0.5
    beta_max = 5.0
    tau = 3.0  # characteristic scale of f_best

    abs_f_best = abs(f_best)
    beta = beta_min + (beta_max - beta_min) * (1.0 - np.exp(-abs_f_best / tau))

    # LCB score for minimization: we want to select the point with the
    # lowest mu - beta*sigma, i.e., maximize -(mu - beta*sigma) = -mu + beta*sigma
    scores = -mu + beta * sigma

    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    # Deadline guard (should be instant, but keep for safety)
    if time.time() > deadline:
        pass

    return scores
# EVOLVE-BLOCK-END
