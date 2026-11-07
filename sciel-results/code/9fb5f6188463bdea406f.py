"""Rewritten acquisition function using cost-aware EI with stage-adaptive exploration."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 295.0  # safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function with stage-adaptive exploration.
    
    Uses Expected Improvement (EI) normalized by cost, with a dynamic
    exploration bonus that transitions from aggressive (LCB-like) early
    to conservative (EI-like) late in the optimization.
    
    Parameters
----------
    mu : array_like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed objective value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        cost_arr = np.asarray(cost, dtype=np.float64)
        cost_arr = np.where(np.isfinite(cost_arr) & (cost_arr > 0), cost_arr, 1.0)
        return 1.0 / cost_arr

    # Ensure arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace non-finite values
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-8)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Calculate Expected Improvement (EI) for minimization
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    
    # Clip z for numerical stability
    z_clipped = np.clip(z, -10, 10)
    
    phi_z = norm.pdf(z_clipped)
    phi_cdf_z = norm.cdf(z_clipped)
    
    # EI calculation
    ei = (f_best - mu) * phi_cdf_z + sigma_safe * phi_z
    ei = np.maximum(ei, 0.0)
    
    # LCB (Lower Confidence Bound) for minimization: mu - beta * sigma
    # This is more exploratory than EI
    # Use a beta that gives a reasonable exploration-exploitation tradeoff
    beta_lcb = 2.0
    lcb = mu - beta_lcb * sigma
    
    # Stage-adaptive: estimate progress based on f_best and typical scale
    # The three-hump-camel function has minimum 0, and values can be up to ~50
    # Use f_best to estimate how far we are: if f_best is close to 0, we're late
    # If f_best is large, we're early
    # Normalize: progress = 1 - exp(-f_best / scale)
    # When f_best is large (early), progress is small -> more exploration
    # When f_best is small (late), progress is large -> more exploitation
    f_best_val = float(f_best) if np.isfinite(f_best) else 10.0
    f_best_clamped = max(f_best_val, 1e-6)
    
    # Scale parameter: the typical range of the objective
    # Three-hump-camel on [-5,5]^2 has values roughly in [0, 50]
    scale = 5.0
    progress = 1.0 - np.exp(-f_best_clamped / scale)
    
    # Smooth transition: weight for EI vs LCB
    # Early (progress ~ 0): mostly LCB (exploration)
    # Late (progress ~ 1): mostly EI (exploitation)
    # Use a smooth sigmoid-like transition
    # weight_ei = progress, weight_lcb = 1 - progress
    # But we want a sharper transition in the middle
    # Use: weight_ei = progress^2 for faster transition to exploitation
    
    weight_ei = progress ** 1.5
    weight_lcb = 1.0 - weight_ei
    
    # Combine EI and LCB
    # Normalize LCB to be comparable to EI
    # LCB can be negative, so shift it
    lcb_shifted = lcb - np.min(lcb)  # make non-negative
    lcb_shifted = np.maximum(lcb_shifted, 0.0)
    
    # Scale LCB to be comparable to EI
    ei_scale = np.max(ei) if np.max(ei) > 0 else 1.0
    lcb_scaled = lcb_shifted * (ei_scale / (np.max(lcb_shifted) + 1e-10))
    
    # Combined acquisition: weighted mix of EI and LCB
    combined = weight_ei * ei + weight_lcb * lcb_scaled
    
    # Cost-aware score
    base_score = combined / cost
    
    # Dynamic exploration bonus: encourages visiting high-uncertainty, low-cost points
    # The bonus should be larger early and smaller late
    # Also scale by the relative uncertainty
    sigma_max = np.max(sigma) if np.max(sigma) > 0 else 1.0
    rel_uncertainty = sigma / (sigma_max + 1e-10)
    
    # Exploration weight: larger early, smaller late
    exploration_weight = 0.1 * (1.0 - progress) + 0.02 * progress
    
    exploration_bonus = exploration_weight * rel_uncertainty * sigma / cost
    
    # Combine scores
    scores = base_score + exploration_bonus
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure non-negative scores for numerical stability
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
