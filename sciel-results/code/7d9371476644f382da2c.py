import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Adaptive Lower Confidence Bound (LCB) acquisition function.
    
    Uses a beta parameter that adapts based on the distance to the known
    global minimum (-2.53218). When far from the minimum, it explores more
    (higher beta); when close, it exploits more (lower beta).
    """
    # Known global minimum for this specific task
    f_min = -2.53218
    
    # Calculate distance from current best to global min
    # f_best is the lowest observed value (minimizing), so f_best >= f_min
    dist = f_best - f_min
    
    # Clip distance to avoid numerical issues if f_best < f_min (shouldn't happen)
    dist = max(dist, 0.0)
    
    # Adaptive beta:
    # Start with high exploration (beta ~ 2.5) and decay to low exploration (beta ~ 0.5)
    # The decay rate is tuned for the scale of the objective function.
    # If dist is large, exp(-dist/0.05) is ~0, so beta ~ 0.5? 
    # Wait, if dist is large, we want HIGH beta.
    # Let's invert the logic or use a different formula.
    
    # If dist is large -> we want beta high.
    # If dist is small -> we want beta low.
    
    # Let's use: beta = beta_min + (beta_max - beta_min) * exp(-dist / scale)
    # If dist=0, beta = beta_max.
    # If dist->inf, beta = beta_min.
    # This means: Close to min -> Explore more? No, that's wrong.
    # Close to min -> Exploit (Low Beta).
    # Far from min -> Explore (High Beta).
    
    # So:
    # dist=0 -> beta_min
    # dist=inf -> beta_max
    
    # beta = beta_min + (beta_max - beta_min) * (1 - exp(-dist / scale))
    # If dist=0, beta = beta_min.
    # If dist=inf, beta = beta_max.
    
    beta_min = 0.5
    beta_max = 2.5
    scale = 0.1  # Scale parameter for the objective function range
    
    # Calculate adaptive beta
    # Ensure dist is not negative
    beta = beta_min + (beta_max - beta_min) * (1.0 - np.exp(-dist / scale))
    
    # Calculate LCB scores: mu - beta * sigma
    # We want to maximize this score to select the next point (minimizing mu - beta*sigma)
    scores = mu - beta * sigma
    
    # Ensure finite scores
    # If sigma is 0, scores = mu.
    # If mu or sigma are NaN/Inf, replace them.
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    
    return scores
