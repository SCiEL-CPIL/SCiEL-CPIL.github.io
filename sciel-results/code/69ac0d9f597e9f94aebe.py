"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds of safety margin

# Cache for adaptive parameters
_adaptive_state = {"call_count": 0}


def _elapsed_time():
    return time.time() - _start_time


def _time_exceeded():
    return _elapsed_time() > (330.0 - _DEADLINE_MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-adjusted acquisition function for Bayesian optimization.
    
    Uses a hybrid Thompson Sampling + LCB approach:
    - Sample a target offset from Gaussian centered at f_best
    - Score candidates by expected improvement toward that offset
    - Cost-aware normalization with uncertainty-adjusted penalty
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions in objective units
    sigma : array-like, shape (2000,)
        GP standard deviation predictions in objective units
    f_best : float
        Best (lowest) observed objective value
    cost : array-like, shape (2000,)
        Evaluation cost for each candidate (1 + mean(x_scaled))
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    _adaptive_state["call_count"] += 1
    
    # Convert to arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Check deadline
    if _time_exceeded():
        # Fallback: return simple cost-normalized negative mu
        return -mu / cost
    
    # Determine phase based on f_best
    # Ackley 2D minimum is 0, max is ~29
    # Use f_best to estimate progress
    f_scale = 25.0
    f_ratio = np.clip(f_best / f_scale, 0.0, 1.0)
    progress = 1.0 - f_ratio  # progress increases as we approach optimum
    
    # Thompson Sampling: sample a target offset
    # The offset represents a "virtual" best value we're trying to beat
    # Early on: offset is close to f_best (exploit current knowledge)
    # Late on: offset can be lower (more aggressive exploitation)
    sigma_max = np.max(sigma)
    if sigma_max < 1e-9:
        sigma_max = 1e-9
    
    # Scale the sampling std by uncertainty and progress
    # Early: smaller std (stay near f_best), Late: larger std (explore lower values)
    sample_std = 0.3 * sigma_max * (0.5 + 0.5 * progress)
    
    # Sample the offset: lower is better (we want to find points that could beat this)
    offset = f_best - np.random.normal(0, sample_std)
    
    # LCB-style score: -mu + beta * sigma
    # Beta adapts: higher early (explore), lower late (exploit)
    beta_min = 0.5
    beta_max = 3.0
    beta = beta_min + (beta_max - beta_min) * f_ratio
    
    lcb_score = -mu + beta * sigma
    
    # Thompson Sampling score: how likely is this point to beat the offset?
    # P(f(x) < offset) = Phi((offset - mu) / sigma)
    # Expected improvement toward offset:
    z_ts = (offset - mu) / sigma
    ei_ts = (offset - mu) * norm.cdf(z_ts) + sigma * norm.pdf(z_ts)
    
    # Combine LCB and Thompson Sampling scores
    # Blend: more LCB early (broad exploration), more TS late (targeted exploitation)
    blend_ts = 0.3 + 0.5 * progress  # blend in [0.3, 0.8]
    
    # Normalize both to comparable scale
    lcb_mean = np.mean(lcb_score)
    lcb_std = np.std(lcb_score)
    if lcb_std < 1e-8:
        lcb_std = 1e-8
    lcb_norm = (lcb_score - lcb_mean) / lcb_std
    
    ts_mean = np.mean(ei_ts)
    ts_std = np.std(ei_ts)
    if ts_std < 1e-8:
        ts_std = 1e-8
    ts_norm = (ei_ts - ts_mean) / ts_std
    
    combined = (1.0 - blend_ts) * lcb_norm + blend_ts * ts_norm
    
    # Cost-awareness: penalty that accounts for cost but is reduced for high uncertainty
    sigma_mean = np.mean(sigma)
    if sigma_mean < 1e-8:
        sigma_mean = 1e-8
    
    # Higher uncertainty points are penalized less by cost (they're worth more exploration)
    alpha = 0.4
    cost_penalty = cost * (1.0 + alpha * sigma / (sigma_mean + 1e-8))
    cost_penalty = np.maximum(cost_penalty, 1e-8)
    
    # Final score: combined score divided by cost penalty
    scores = combined / cost_penalty
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
