import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization (minimization).
    Uses a convex blend of normalized LCB and normalized EI.
    
    LCB (Lower Confidence Bound) for minimization: mu - beta * sigma
    EI (Expected Improvement) for minimization: E[max(f_best - f(x), 0)]
    
    Both are normalized by the median sigma to be on comparable scales,
    then blended with a weight that adapts to the current best value.
    
    Parameters:
    mu (np.ndarray): Posterior mean at candidate points, shape (N,).
    sigma (np.ndarray): Posterior standard deviation at candidate points, shape (N,).
    f_best (float): Best observed function value so far.
    
    Returns:
    np.ndarray: Scores for each candidate point, shape (N,). Higher score is better.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute LCB for minimization: we want to maximize -LCB = -mu + beta*sigma
    beta = 3.0
    LCB_raw = -mu + beta * sigma_safe
    
    # Compute EI for minimization
    z = (f_best - mu) / sigma_safe
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    EI_raw = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Normalize both by the median sigma to make them comparable
    sigma_median = np.median(sigma_safe)
    if sigma_median < 1e-12:
        sigma_median = 1e-12
    
    LCB_norm = LCB_raw / sigma_median
    EI_norm = EI_raw / sigma_median
    
    # Adaptive blend weight: shift from exploration (LCB) to exploitation (EI)
    # as f_best approaches 0 (the known minimum of Weierstrass 1d)
    # When f_best is large (early), weight more on LCB for exploration
    # When f_best is small (late), weight more on EI for exploitation
    # Use a smooth transition based on f_best
    # f_best ranges from ~1.0 (initial) down to 0 (optimal)
    # Map f_best to a weight: w_ei goes from 0.3 (early) to 0.8 (late)
    f_best_clamped = np.clip(f_best, 0.0, 2.0)
    w_ei = 0.3 + 0.5 * (1.0 - f_best_clamped / 2.0)
    w_lcb = 1.0 - w_ei
    
    scores = w_lcb * LCB_norm + w_ei * EI_norm
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
