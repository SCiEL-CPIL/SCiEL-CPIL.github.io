"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses a hybrid approach combining:
    1. Lower Confidence Bound (LCB) for exploration
    2. Expected Improvement (EI) for exploitation
    3. Adaptive weighting based on confidence
    
    Args:
        mu: array of shape (2000,) - posterior mean
        sigma: array of shape (2000,) - posterior standard deviation
        f_best: float - best observed value so far
    
    Returns:
        scores: array of shape (2000,) - acquisition scores (higher = better)
    """
    # Track time to ensure we don't exceed budget
    t_start = time.time()
    DEADLINE = 325.0  # Safety margin before 330s kill
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Number of evaluations so far (approximate from sigma pattern)
    # When we have fewer observations, sigma is generally higher
    # Use the mean sigma as a proxy for exploration level
    mean_sigma = np.mean(sigma_safe)
    
    # Adaptive beta: start with more exploration, reduce over time
    # Beta decays as we get more confident
    beta_base = 2.5
    beta_decay = 0.5
    # Use a simple heuristic: lower mean_sigma means more observations
    # Normalize mean_sigma to [0, 1] roughly
    sigma_norm = np.clip(mean_sigma / (np.max(sigma_safe) + 1e-10), 0.0, 1.0)
    beta = beta_base * (1.0 - beta_decay * (1.0 - sigma_norm))
    
    # 1. LCB score for minimization: we want to minimize mu - beta * sigma
    # So the acquisition score (higher is better) is: -mu + beta * sigma
    lcb_score = -mu + beta * sigma_safe
    
    # 2. Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Use scipy's norm.cdf and norm.pdf, but compute manually for speed
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # phi(z) = (1/sqrt(2*pi)) * exp(-z^2/2)
    sqrt_2pi = np.sqrt(2.0 * np.pi)
    phi_z = np.exp(-0.5 * z_clipped ** 2) / sqrt_2pi
    # Use erf for CDF
    from scipy.special import erf
    Phi_z = 0.5 * (1.0 + erf(z_clipped / np.sqrt(2.0)))
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # 3. Normalize both scores to have zero mean and unit variance
    # This makes the combination robust to scale differences
    
    lcb_mean = np.mean(lcb_score)
    lcb_std = np.std(lcb_score)
    if lcb_std < 1e-9:
        z_lcb = np.zeros_like(lcb_score)
    else:
        z_lcb = (lcb_score - lcb_mean) / lcb_std
    
    ei_mean = np.mean(ei)
    ei_std = np.std(ei)
    if ei_std < 1e-9:
        z_ei = np.zeros_like(ei)
    else:
        z_ei = (ei - ei_mean) / ei_std
    
    # 4. Combine with adaptive weights
    # When uncertainty is high (early), weight LCB more
    # When uncertainty is low (late), weight EI more
    # Use sigma_norm as proxy: high sigma_norm = early = more exploration
    w_explore = 0.4 + 0.4 * sigma_norm  # ranges from 0.4 to 0.8
    w_exploit = 1.0 - w_explore
    
    scores = w_explore * z_lcb + w_exploit * z_ei
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Deadline guard
    if time.time() > DEADLINE:
        # Return the best result we have
        pass
    
    return scores
# EVOLVE-BLOCK-END
