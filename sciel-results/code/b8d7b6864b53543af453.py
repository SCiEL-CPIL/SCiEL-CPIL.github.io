"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid of Expected Improvement (EI) and
    Lower Confidence Bound (LCB). This structural change combines the
    exploitation strength of EI with the exploration of LCB, which is
    better suited for multimodal landscapes like Hartmann 6D than pure
    LCB.
    
    The hybrid score is: alpha * EI + (1 - alpha) * LCB
    where EI is normalized and LCB is the standard GP-LCB for minimization.
    """
    # ----- time guard (safety margin) -----
    t0 = time.time()
    deadline = t0 + 295.0  # leave 5 s for caller overhead

    n = mu.shape[0]

    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-12)

    # Compute Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi and phi are the CDF and PDF of the standard normal distribution.
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # EI is non-negative; higher is better

    # Compute LCB for minimization
    # LCB = -mu - kappa * sigma
    # Higher LCB score means more attractive (lower predicted value + exploration)
    kappa = 2.5
    lcb = -mu - kappa * sigma_safe

    # Normalize both to [0, 1] range for blending
    ei_max = np.max(ei)
    ei_min = np.min(ei)
    if ei_max > ei_min:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)

    lcb_max = np.max(lcb)
    lcb_min = np.min(lcb)
    if lcb_max > lcb_min:
        lcb_norm = (lcb - lcb_min) / (lcb_max - lcb_min)
    else:
        lcb_norm = np.zeros_like(lcb)

    # Blend: 50% EI, 50% LCB
    alpha = 0.5
    scores = alpha * ei_norm + (1 - alpha) * lcb_norm

    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # If we are dangerously close to the deadline, return immediately
    if time.time() > deadline - 0.5:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
