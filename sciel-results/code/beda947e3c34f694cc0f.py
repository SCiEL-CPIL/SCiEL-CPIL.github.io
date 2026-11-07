import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Thompson Sampling with exploitation bias.
    
    Parameters
    ----------
    mu : array_like, shape (n,)
        Posterior mean predictions at candidate points.
    sigma : array_like, shape (n,)
        Posterior standard deviation predictions at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: Sample from posterior predictive distribution
    # y ~ N(mu, sigma^2)
    z = np.random.normal(0, 1, size=mu.shape)
    y_samples = mu + sigma_safe * z
    
    # Base score: higher is better. Since we minimize y, score is -y.
    base_score = -y_samples
    
    # Deterministic exploitation bias toward the global minimum (f_best)
    # This nudges the score based on the deterministic mean prediction relative to f_best.
    # If mu is much lower than f_best, this term is positive and boosts the score,
    # counteracting an unlucky high sample (z >> 0).
    bias_strength = 0.1
    exploitation_bias = bias_strength * (f_best - mu)
    
    scores = base_score + exploitation_bias
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
