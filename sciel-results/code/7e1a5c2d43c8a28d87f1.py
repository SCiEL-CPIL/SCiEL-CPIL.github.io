"""Cost-aware acquisition function with adaptive hybrid for Powell 4D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 3.0  # seconds before the hard 330s limit


def _elapsed():
    return time.time() - _START_TIME


def _deadline_exceeded():
    return _elapsed() > (330.0 - _DEADLINE_MARGIN)


def _compute_poi_adaptive(mu, sigma, f_best):
    """
    Compute adaptive Probability of Improvement with cost correction.
    
    POI = Phi((f_best - mu) / sigma)
    We add a small exploration bonus based on the PDF to encourage
    evaluating points at the frontier of the current best value.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10.0, 10.0)
    
    poi = norm.cdf(z_clipped)
    pdf = norm.pdf(z_clipped)
    
    # Blend: primarily POI, but add exploration weight from PDF
    # This encourages points where improvement is likely but also
    # where there's significant uncertainty at the frontier
    adaptive_poi = 0.85 * poi + 0.15 * pdf * 10.0  # Scale PDF to be comparable
    return adaptive_poi


def _compute_expected_improvement_normalized(mu, sigma, f_best):
    """
    Compute normalized Expected Improvement.
    
    EI = sigma * (z * Phi(z) + phi(z))
    We normalize by sigma to get a scale-invariant measure.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10.0, 10.0)
    
    cdf = norm.cdf(z_clipped)
    pdf = norm.pdf(z_clipped)
    
    # Normalized EI: divide by sigma to get scale-invariant
    # This is (z * Phi(z) + phi(z))
    ei_normalized = z_clipped * cdf + pdf
    
    return ei_normalized


def _compute_cost_aware_ucb(mu, sigma, f_best, cost, beta=2.0):
    """
    Compute cost-aware UCB with adaptive beta.
    
    UCB = -mu + beta * sigma
    Cost-aware: divide by cost
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    ucb = -mu + beta * sigma_safe
    return ucb / cost


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (powell 4d).

    Uses a hybrid of normalized EI, adaptive POI, and UCB with cost-awareness.
    The blend adapts based on the uncertainty level to balance exploitation
    and exploration.

    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed function value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate, shape (2000,).

    Returns
    -------
    np.ndarray, shape (2000,)
        Finite scores; higher = prefer to evaluate next.
    """
    # Explicit deadline guard
    if _deadline_exceeded():
        # Return a safe, finite fallback
        sigma_safe = np.maximum(sigma, 1e-9)
        # Fallback to simple cost-aware LCB
        score = (-mu + 2.0 * sigma_safe) / (cost + 1e-9)
        return np.where(np.isfinite(score), score, -1e10)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against non-finite or invalid values
    mu = np.where(np.isfinite(mu), mu, 1e10)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    sigma_safe = np.maximum(sigma, 1e-9)

    # Compute each component
    poi_adaptive = _compute_poi_adaptive(mu, sigma, f_best)
    ei_normalized = _compute_expected_improvement_normalized(mu, sigma, f_best)
    ucb_cost_aware = _compute_cost_aware_ucb(mu, sigma, f_best, cost, beta=2.0)

    # Normalize each component to [0, 1] range for stable blending
    # POI is already in [0, 1]
    # EI normalized can be negative; shift and scale
    ei_min = np.min(ei_normalized)
    ei_max = np.max(ei_normalized)
    ei_range = ei_max - ei_min + 1e-9
    ei_norm_scaled = (ei_normalized - ei_min) / ei_range
    
    # UCB cost-aware: normalize
    ucb_min = np.min(ucb_cost_aware)
    ucb_max = np.max(ucb_cost_aware)
    ucb_range = ucb_max - ucb_min + 1e-9
    ucb_norm_scaled = (ucb_cost_aware - ucb_min) / ucb_range

    # Adaptive blending based on uncertainty
    # When sigma is high (early), weight more toward exploration (UCB, POI)
    # When sigma is low (later), weight more toward exploitation (EI)
    mean_sigma = np.mean(sigma_safe)
    # Use a smooth transition: t in [0, 1], 0 = high uncertainty, 1 = low uncertainty
    # Assume sigma ranges from ~0.1 (low) to ~1.0 (high) in objective units
    t = np.clip(1.0 - (mean_sigma - 0.1) / 0.9, 0.0, 1.0)
    
    # Weights: when t is high (low uncertainty), favor EI
    # When t is low (high uncertainty), favor POI and UCB
    w_ei = 0.4 + 0.4 * t      # 0.4 to 0.8
    w_poi = 0.3 - 0.1 * t     # 0.3 to 0.2
    w_ucb = 0.3 - 0.3 * t     # 0.3 to 0.0
    
    # Normalize weights to sum to 1
    w_sum = w_ei + w_poi + w_ucb
    w_ei /= w_sum
    w_poi /= w_sum
    w_ucb /= w_sum

    # Combined score
    score = w_ei * ei_norm_scaled + w_poi * poi_adaptive + w_ucb * ucb_norm_scaled

    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -1e10)

    return score
# EVOLVE-BLOCK-END
