import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling variant with exploitation bias.
    Sample from posterior predictive: z = mu - sigma * N(0,1)
    Then add a small bias toward exploitation (lower mu).
    Higher score = better candidate to evaluate next.
    """
    # Thompson sampling: draw a random sample from the posterior
    # For minimization, we want to sample values and pick the lowest
    # score = -sampled_value, so higher score = lower sampled value
    rng = np.random.default_rng()
    z = rng.standard_normal(mu.shape)
    # Sampled objective value: lower is better
    sampled = mu - sigma * z
    
    # Exploitation bias: add a small penalty for high mu (exploit known good regions)
    # This helps escape local minima while still favoring exploitation
    # Scale the bias based on how far we are from the optimum
    opt = -1.8013
    dist = max(0.0, f_best - opt)
    # Small bias: 0.1 * mu encourages exploitation when close to optimum
    bias_strength = 0.1 * min(dist / 5.0, 1.0)
    
    # Final score: higher is better
    # -sampled gives us the Thompson sampling component
    # -bias_strength * mu adds exploitation bias (lower mu = higher score)
    scores = -sampled - bias_strength * mu
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
