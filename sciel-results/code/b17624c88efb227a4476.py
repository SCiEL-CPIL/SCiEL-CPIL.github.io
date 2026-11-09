"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Pareto-based multi-objective acquisition: weighted sum of normalized EI
    and normalized predictive uncertainty, with dynamic weight that shifts
    from exploration to exploitation as evaluations accumulate.
    
    mu: (n,) array of GP mean predictions
    sigma: (n,) array of GP standard deviation predictions
    f_best: current best observed value (minimum)
    
    Returns: (n,) array of scores, higher is better
    """
    start_time = time.time()
    deadline = start_time + 295.0

    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)

    # Expected Improvement for minimization: EI = (f_best - mu) * Phi((f_best - mu)/sigma)
    # + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues in extreme tails
    z = np.clip(z, -10, 10)
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    EI = (f_best - mu) * Phi + sigma_safe * phi
    EI = np.maximum(EI, 0.0)

    # Normalize EI to [0, 1]
    ei_max = np.max(EI)
    if ei_max > 1e-12:
        EI_norm = EI / ei_max
    else:
        EI_norm = np.zeros_like(EI)

    # Normalize sigma to [0, 1]
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-12:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)

    # Dynamic weight: w increases with number of evaluations (proxy via f_best relative scale)
    # Early: more exploration (higher weight on sigma), Late: more exploitation (higher weight on EI)
    # Use a smooth transition. f_best starts near 0 (Hartmann 6d range is roughly [-3.3, 0.5])
    # We estimate evaluation count indirectly: early iterations have higher sigma relative to scale
    # Use sigma_max as a proxy for remaining uncertainty / early stage
    # When sigma_max is large relative to the objective scale, we're early -> explore more
    # Objective scale for Hartmann 6d: range is about 3.8, so use that as reference
    obj_scale = 3.8
    sigma_ratio = np.clip(sigma_max / obj_scale, 0.0, 1.0)
    
    # Weight on exploration: high when sigma_ratio is high (early), low when low (late)
    # w_explore = 1 - sigma_ratio would be backwards; we want MORE exploration early
    # Actually: early = high sigma = want to explore = high weight on sigma
    # Late = low sigma = want to exploit = high weight on EI
    # So: w_sigma = 0.3 + 0.7 * sigma_ratio  (more uncertainty -> more exploration weight)
    #     w_ei    = 1.0 - w_sigma
    w_sigma = 0.3 + 0.7 * sigma_ratio
    w_ei = 1.0 - w_sigma

    # Combined score: weighted sum of normalized objectives
    scores = w_ei * EI_norm + w_sigma * sigma_norm

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        return scores

    return scores
# EVOLVE-BLOCK-END
