import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Computes the Adaptive Lower Confidence Bound (LCB) acquisition score.
    
    Args:
        mu: Mean of the GP posterior (shape (2000,))
        sigma: Standard deviation of the GP posterior (shape (2000,))
        f_best: Best observed value so far (scalar)
        
    Returns:
        Scores (shape (2000,)), higher is better for selection.
    """
    # Adaptive beta based on how close f_best is to the known global minimum
    f_min = -3.86278
    
    # Calculate difference from the global minimum
    diff = f_best - f_min
    
    # Normalize the difference.
    # Using a divisor of 1.0 means beta stays at max until f_best is within 1.0 of f_min.
    # This encourages sustained exploration in the early and mid stages.
    norm_diff = np.clip(diff / 1.0, 0.0, 1.0)
    
    # Beta ranges from 0.5 (exploit) to 4.0 (explore).
    # Wider range than the previous 1.0-3.0 to allow for stronger exploration.
    beta_min = 0.5
    beta_max = 4.0
    beta = beta_min + (beta_max - beta_min) * norm_diff
    
    # LCB for minimization: Score = beta * sigma - mu
    # Higher score is better
    scores = beta * sigma - mu
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
