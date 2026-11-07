import numpy as np

def acquisition(mu, sigma, f_best):
    # UCB acquisition function for minimization
    # We want to maximize the score, so we negate the mean
    # and add the exploration term.
    
    # Adaptive kappa: scale exploration based on uncertainty
    # When sigma is high, we explore more; when it's low, we exploit
    sigma_mean = np.mean(sigma)
    sigma_max = np.max(sigma)
    
    # Use a smooth scaling: kappa decreases as we get more certain
    # Start with higher exploration, decay towards exploitation
    if sigma_max > 0:
        # Normalize sigma to [0, 1] range
        sigma_norm = sigma / sigma_max
        # Adaptive kappa: higher when uncertain, lower when certain
        # Using a smooth transition based on mean uncertainty
        base_kappa = 2.0  # Start with more exploration
        decay_factor = 0.5  # How quickly to reduce exploration
        kappa = base_kappa * np.exp(-decay_factor * (1 - sigma_mean))
    else:
        kappa = 0.0
    
    score = -mu + kappa * sigma
    return score
