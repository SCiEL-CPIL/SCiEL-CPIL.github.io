"""Cost-aware acquisition function for Bayesian optimization on rastrigin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 280.0
_start_time = time.time()


def _check_deadline():
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Regret and uncertainty-driven exploration.
    
    Uses a hybrid approach:
    1. Expected Regret (ER) for direct objective improvement per unit cost
    2. Uncertainty-weighted exploration term to handle multimodal Rastrigin landscape
    3. Adaptive weighting based on current confidence (sigma relative to f_best)
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled)).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma and cost are strictly positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute Z-score for Expected Regret
    Z = (f_best - mu) / sigma_safe
    
    # Compute Phi(Z) and phi(Z) efficiently
    # Using scipy.stats.norm for accuracy
    from scipy.stats import norm
    Phi = norm.cdf(Z)
    phi = norm.pdf(Z)
    
    # Expected Regret: E[max(0, f_best - f(x))]
    # For minimization, this is the expected improvement in best value
    ER = (f_best - mu) * Phi + sigma_safe * phi
    ER = np.maximum(ER, 0.0)
    
    # Exploration term: emphasize high uncertainty regions
    # Use sigma directly as exploration signal, but normalized
    # to be comparable with ER scale
    # Scale sigma by typical magnitude of ER to balance terms
    er_scale = np.mean(ER[ER > 0]) if np.any(ER > 0) else max(abs(f_best), 1.0)
    exploration = sigma_safe * np.log1p(sigma_safe / (er_scale + 1e-9))
    
    # Adaptive weighting: when we're confident (low sigma), trust ER more
    # When uncertain (high sigma), weight exploration more
    # Use a sigmoid-like transition based on sigma relative to er_scale
    sigma_norm = sigma_safe / (er_scale + 1e-9)
    # Weight for ER: high when sigma is low
    w_er = 1.0 / (1.0 + sigma_norm)
    # Weight for exploration: high when sigma is high
    w_exp = sigma_norm / (1.0 + sigma_norm)
    
    # Combine: weighted sum of ER and exploration
    combined = w_er * ER + w_exp * exploration
    
    # Normalize by cost
    scores = combined / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
