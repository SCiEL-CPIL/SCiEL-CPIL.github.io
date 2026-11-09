"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Quantile-based acquisition for Bayesian optimization of sphere 1D.
    
    Structural change: Instead of a linear confidence bound (UCB/LCB), we use
    an adaptive quantile of the posterior. The score is the negative of the
    q-quantile of the posterior predictive, where q adapts based on f_best.
    
    Early (f_best large): q is low (e.g., 0.05) -> very aggressive exploitation
    Late (f_best small): q is higher (e.g., 0.3) -> more balanced
    
    This is a different modelling family: quantile optimization vs. expected value.
    """
    # Deadline guard: track elapsed time
    import time
    _t0 = time.time()
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-10)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # f_best_safe to avoid issues
    f_best_safe = max(float(f_best), 1e-10)
    
    # For sphere on [-5,5], f ranges from 0 to 25
    # Normalize f_best to [0, 1] where 1 = fully exploratory (early), 0 = fully exploitative (late)
    f_max = 25.0
    f_norm = min(f_best_safe / f_max, 1.0)
    
    # Adaptive quantile: 
    # When f_best is large (early), we want to explore -> higher quantile of f (less aggressive)
    # When f_best is small (late), we want to exploit -> lower quantile of f (more aggressive)
    # 
    # We pick the quantile q such that we're looking at the lower tail of the posterior.
    # score = -Q_q(f(x)) where Q_q is the q-quantile.
    # Lower q = more aggressive (picking lower values)
    #
    # q ranges from 0.01 (very aggressive, late) to 0.5 (balanced, early)
    # Actually, let's think about this differently:
    # We want to maximize P(f(x) < f_best) weighted by improvement, but using a quantile.
    # 
    # Alternative: score = -mu + z_q * sigma where z_q is the z-score for quantile q.
    # This is still a linear bound but with a different interpretation.
    #
    # Let's use a true quantile approach:
    # The q-quantile of N(mu, sigma^2) is mu + z_q * sigma
    # where z_q = norm.ppf(q)
    #
    # For minimization, we want the lower quantile.
    # q = 0.1 means we're looking at the 10th percentile -> aggressive
    # q = 0.5 means median -> less aggressive
    
    # Adapt q: early -> q closer to 0.5 (explore), late -> q closer to 0.05 (exploit)
    # Use a smooth transition
    q_min = 0.02   # very aggressive
    q_max = 0.5    # balanced
    
    # Transition: when f_norm is high (early), q is high; when low (late), q is low
    # Use quadratic for sharper transition to exploitation
    q = q_min + (q_max - q_min) * (f_norm ** 2)
    
    # Compute the z-score for this quantile
    z_q = norm.ppf(q)
    
    # The q-quantile of the posterior at each point
    # quantile = mu + z_q * sigma
    # We want to MINIMIZE the function, so we score higher for LOWER quantiles
    # score = -(mu + z_q * sigma) = -mu - z_q * sigma
    # Since z_q is negative (q < 0.5), -z_q is positive, so this adds exploration
    
    scores = -mu - z_q * sigma
    
    # Deadline guard
    if time.time() - _t0 > 280:
        # Return best found so far (all computed)
        pass
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
