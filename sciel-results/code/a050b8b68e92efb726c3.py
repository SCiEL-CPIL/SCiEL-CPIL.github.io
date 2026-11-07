import numpy as np

def acquisition(mu, sigma, f_best):
    # Thompson Sampling approximation
    # Sample a single noise vector for all candidates?
    # Or independent noise for each?
    # Standard TS samples one function, so the noise is correlated across points?
    # No, TS samples a function value for each point?
    # Actually, TS samples a function from the GP.
    # The values at the 2000 points are correlated.
    # We cannot easily sample the full GP posterior here because we don't have the kernel matrix or the training data.
    # We only have mu and sigma for the candidates.
    # So we can only assume independence between the candidates' noises for the approximation.
    # This is "Independent Thompson Sampling" or just adding noise.
    
    # If we add independent noise:
    # score_i = -mu_i + sigma_i * z_i
    # This encourages exploring points with high sigma.
    
    z = np.random.normal(0, 1, size=mu.shape)
    scores = -mu + sigma * z
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
