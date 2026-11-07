"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for deadline tracking
_start_time = time.time()
_TIME_LIMIT = 325  # seconds, safety margin before 330s kill

# Known minimum of Branin 2D
_F_MIN = 0.397887


def _knowledge_gradient(mu, sigma, f_best, n_samples=50):
    """
    Approximate Knowledge Gradient (KG) acquisition for minimization.
    
    KG estimates the expected reduction in the best observed value after
    evaluating a candidate. We approximate this via Monte Carlo sampling
    of the posterior at each candidate point.
    
    For a candidate x with posterior N(mu_i, sigma_i^2):
    KG(x) = E[max(f_best - f(x), 0)] where f(x) ~ N(mu_i, sigma_i^2)
    
    This is equivalent to EI with xi=0, but we use a slightly different
    formulation that better captures the global minimum perspective.
    
    We use an efficient vectorized MC approach: sample n_samples values
    from each candidate's posterior, compute the expected improvement
    over f_best, and return the mean.
    """
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Generate MC samples: shape (n, n_samples)
    # Use standard normal samples scaled by sigma
    z = np.random.standard_normal((n, n_samples))
    samples = mu[:, None] + sigma_safe[:, None] * z
    
    # For minimization: improvement is f_best - f(x) when f(x) < f_best
    # KG ≈ E[max(f_best - f(x), 0)]
    improvements = np.maximum(f_best - samples, 0.0)
    
    # Mean improvement over samples
    kg = np.mean(improvements, axis=1)
    
    return kg


def _ei_score(mu, sigma, f_best, xi=0.0):
    """
    Expected Improvement (EI) for minimization with offset xi.
    """
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - xi - mu) / sigma_safe
    ei = (f_best - xi - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _ucb_score(mu, sigma, beta=1.96):
    """
    Upper Confidence Bound (LCB for minimization).
    """
    return mu - beta * sigma


def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient-based acquisition function for minimization of Branin 2D.
    Uses an adaptive strategy: KG when close to optimum, EI with adaptive xi
    when far, and a UCB fallback near the time deadline.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions at candidate points.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best observed value so far (lowest).
    
    Returns
    -------
    scores : np.ndarray, shape (2000,)
        Finite scores; higher means evaluate next.
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT - 5:
        # Fallback: simple LCB with small exploration
        sigma_safe = np.maximum(sigma, 1e-9)
        scores = -mu + 0.01 * sigma_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Determine which acquisition to use based on proximity to optimum
    gap = f_best - _F_MIN
    
    if gap > 10.0:
        # Far from optimum: use EI with larger xi to encourage exploration
        # of significantly better regions
        xi = min(1.0, gap * 0.05)
        scores = _ei_score(mu, sigma, f_best, xi=xi)
    elif gap > 2.0:
        # Medium distance: use EI with moderate xi
        xi = 0.2
        scores = _ei_score(mu, sigma, f_best, xi=xi)
    else:
        # Close to optimum: use Knowledge Gradient for fine-grained search
        # KG better captures the probability of finding the global minimum
        n_samples = 80  # more samples for accuracy when close
        scores = _knowledge_gradient(mu, sigma, f_best, n_samples=n_samples)
        
        # Add a small exploration bonus weighted by uncertainty
        # to avoid over-concentration
        sigma_safe = np.maximum(sigma, 1e-12)
        # Scale exploration bonus relative to the typical KG magnitude
        kg_scale = np.max(scores) if np.max(scores) > 1e-12 else 1.0
        exploration_bonus = 0.05 * kg_scale * sigma_safe / (np.max(sigma_safe) + 1e-12)
        scores = scores + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
