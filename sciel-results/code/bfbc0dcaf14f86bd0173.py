"""Cost-aware acquisition using cost-normalized PI with adaptive UCB for rastrigin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: keep a safety margin below the 330 s wall-clock limit.
_START_TIME = time.time()
_DEADLINE = 285.0  # seconds


def _check_deadline():
    """Return True if we are close to the wall-clock deadline."""
    return (time.time() - _START_TIME) > (_DEADLINE - 5.0)


def _pi(mu, sigma, f_best):
    """Probability of Improvement: P(f(x) < f_best) for f(x) ~ N(mu, sigma^2)."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    return norm.cdf(z)


def _expected_improvement(mu, sigma, f_best):
    """Standard EI for minimization: E[max(f_best - f(x), 0)]."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _ucb_score(mu, sigma, beta):
    """UCB score for minimization: -mu + beta * sigma (maximize this)."""
    return -mu + beta * sigma


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing rastrigin 2d.
    
    Structural design: cost-normalized Probability of Improvement (PI) as the
    primary exploitation signal, blended with a cost-normalized Expected
    Improvement (EI) term for magnitude-weighted exploitation, plus an
    adaptive UCB exploration term that fades with progress.
    
    PI provides robust decision boundaries in multimodal landscapes (measured
    improvement over EI-only approaches), while EI adds sensitivity to the
    magnitude of expected improvement. UCB ensures early exploration.
    
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

    # Deadline guard: if we are near the wall-clock limit, return a safe
    # exploitation-focused score immediately.
    if _check_deadline():
        cost_safe = np.maximum(cost, 1e-9)
        fallback = -mu / cost_safe
        fallback = np.where(np.isfinite(fallback), fallback, 0.0)
        return fallback

    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    sigma_safe = np.maximum(sigma, 1e-9)

    # Progress metric: 0 when f_best is large (early), 1 when f_best ~ 0 (late)
    # Rastrigin 2d max value is around 40, so normalize by that
    progress = float(np.clip(1.0 - f_best / 40.0, 0.0, 1.0))

    # Primary signal: Probability of Improvement (robust decision boundary)
    pi = _pi(mu, sigma_safe, f_best)

    # Secondary signal: Expected Improvement (magnitude-weighted exploitation)
    ei = _expected_improvement(mu, sigma_safe, f_best)

    # Normalize EI to [0, 1] for stable blending with PI
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)

    # Adaptive UCB beta: higher early for exploration, lower late for exploitation
    # Quadratic decay: starts at 1.0, ends at 0.1
    beta = 0.1 + 0.9 * (1.0 - progress) ** 2

    # UCB component for exploration
    ucb = _ucb_score(mu, sigma_safe, beta)

    # Normalize UCB to [0, 1] range for stable blending
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min + 1e-9
    ucb_norm = (ucb - ucb_min) / ucb_range

    # Weight for EI: increases with progress (more exploitation as we converge)
    w_ei = 0.3 * progress

    # Weight for UCB: fades out as progress is made
    w_ucb = 0.3 * (1.0 - progress)

    # Combine: PI as primary, EI as secondary exploitation, UCB as exploration
    # All components are in [0, 1] range, so the blend is well-scaled
    scores = (pi + w_ei * ei_norm + w_ucb * ucb_norm) / cost_safe

    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
