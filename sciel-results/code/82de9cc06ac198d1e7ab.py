import numpy as np

def acquisition(mu, sigma, f_best):
    # Adaptive UCB acquisition function for minimization
    # Use a kappa that adapts based on the scale of uncertainty
    # relative to the current best value.
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Compute a reference scale: the standard deviation of sigma
    # This helps normalize the exploration term
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma)
    
    # Adaptive kappa: larger when uncertainty is high relative to best
    # This encourages exploration early on and exploitation later
    # Base kappa around 1.0, modulated by the uncertainty landscape
    if sigma_mean > 1e-10:
        # Normalize by the range of sigma to get a shape factor
        sigma_range = np.ptp(sigma)
        if sigma_range > 1e-10:
            # High variance in sigma suggests diverse uncertainty -> explore more
            exploration_factor = 1.0 + 0.5 * (sigma_std / sigma_range)
        else:
            exploration_factor = 1.0
    else:
        exploration_factor = 0.5
    
    # Also consider how far we are from the minimum (Ackley min is 0)
    # If f_best is close to 0, reduce exploration
    if f_best < 0.1:
        exploration_factor *= 0.7
    
    kappa = exploration_factor
    
    score = -mu + kappa * sigma
    return score
