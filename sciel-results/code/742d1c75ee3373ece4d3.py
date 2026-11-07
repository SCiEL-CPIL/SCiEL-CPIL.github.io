import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) with adaptive exploration
    for minimization of Ackley 2D.
    
    Parameters:
    mu (np.ndarray): Posterior mean, shape (2000,).
    sigma (np.ndarray): Posterior standard deviation, shape (2000,).
    f_best (float): Best observed value so far.
    
    Returns:
    np.ndarray: Scores for the 2000 candidates, higher is better.
    """
    t_start = time.time()
    deadline = t_start + 250.0
    
    # Safe sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardize for stable computation
    # Use a reference scale based on the problem domain
    # Ackley 2D range is roughly [0, ~27.6]
    # But we work in objective units where mu and sigma are already in those units
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # Phi is standard normal CDF, phi is standard normal PDF
    
    from scipy.stats import norm
    
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    # EI for minimization: improvement is f_best - mu (positive when mu < f_best)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # EI is always >= 0, higher means more expected improvement
    # But we also want to encourage exploration in early stages
    
    # Adaptive exploration: add a UCB-like term that decays as we find better solutions
    # The scale of the problem: Ackley 2D max is about 27.6, min is 0
    # f_best starts around ~10-15 and decreases toward 0
    
    # Exploration weight: higher when f_best is far from optimum (0)
    # This encourages more exploration early, less later
    exploration_weight = np.maximum(0.5 * (1.0 - f_best / 27.6), 0.1)
    
    # UCB component for exploration
    ucb_component = exploration_weight * sigma_safe
    
    # Combine EI and UCB
    # EI handles exploitation, UCB handles exploration
    scores = ei + ucb_component
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
