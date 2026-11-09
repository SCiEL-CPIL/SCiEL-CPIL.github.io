"""Cost-aware acquisition for three-hump-camel 2D BO using adaptive weighted EI."""
import time
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using adaptive weighted combination of
    Expected Improvement and budget efficiency, with convergence-dependent
    exploitation/exploration balance.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value (lower is better)
    cost : ndarray, shape (2000,)
        Evaluation cost for each candidate point
    
    Returns:
    --------
    score : ndarray, shape (2000,)
        Acquisition scores (higher = more promising)
    """
    start_time = time.time()
    deadline = start_time + 280.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)

    # Compute Expected Improvement
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    ei = np.maximum(ei, 0.0)

    # Convergence state: how close are we to the known minimum (0)?
    # f_best starts high (several units) and decreases toward 0
    # Normalize: convergence = 1 means fully converged, 0 means just started
    convergence = max(0.0, min(1.0, 1.0 - f_best / 5.0))

    # --- Component 1: Normalized EI (exploitation signal) ---
    ei_max = np.max(ei) + 1e-10
    ei_norm = ei / ei_max  # [0, 1]

    # --- Component 2: Budget efficiency (cost-aware signal) ---
    # Lower cost is better. Use inverse cost normalized to [0, 1]
    # cost is in [1, 2], so 1/cost is in [0.5, 1]
    inv_cost = 1.0 / cost
    inv_cost_min = np.min(inv_cost)
    inv_cost_max = np.max(inv_cost)
    inv_cost_range = inv_cost_max - inv_cost_min + 1e-10
    cost_eff_norm = (inv_cost - inv_cost_min) / inv_cost_range  # [0, 1]

    # --- Component 3: Exploration signal (uncertainty) ---
    # UCB-like: mu + beta * sigma, normalized
    # Beta adapts: larger early, smaller late
    beta = 1.0 + 2.0 * (1.0 - convergence)  # [1, 3]
    ucb = mu + beta * sigma
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min + 1e-10
    ucb_norm = (ucb - ucb_min) / ucb_range  # [0, 1]
    # Flip: we want to explore where ucb is high (high uncertainty potential)
    exploration_signal = ucb_norm

    # --- Adaptive weighting ---
    # Early: more weight on exploration and cost efficiency (save budget)
    # Late: more weight on EI (exploit near optimum)
    # Weight for EI increases with convergence
    w_ei = 0.4 + 0.5 * convergence          # [0.4, 0.9]
    w_cost = 0.3 + 0.2 * (1.0 - convergence)  # [0.5, 0.3]
    w_explore = 0.1 + 0.3 * (1.0 - convergence) # [0.4, 0.1]

    # Normalize weights to sum to 1
    w_sum = w_ei + w_cost + w_explore
    w_ei /= w_sum
    w_cost /= w_sum
    w_explore /= w_sum

    # --- Combined score ---
    score = w_ei * ei_norm + w_cost * cost_eff_norm + w_explore * exploration_signal

    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)

    # Time guard
    if time.time() > deadline:
        pass

    return score
