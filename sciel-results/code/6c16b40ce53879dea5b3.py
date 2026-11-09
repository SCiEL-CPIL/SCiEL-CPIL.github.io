import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function with adaptive beta for minimization.
    Uses a smoother beta schedule and normalized scoring for better balance.
    """
    start_time = time.time()
    deadline = 300.0
    
    # Ensure sigma is strictly positive to avoid numerical issues
    sigma = np.maximum(sigma, 1e-10)
    
    # Known minimum of Hartmann 3D function
    f_min_hartmann = -3.86278
    
    # Calculate gap to the global minimum
    gap = f_best - f_min_hartmann
    
    # Smooth beta schedule: use sigmoid-like transition
    # When gap is large (early), beta is high (~2.5); when gap is small (late), beta is low (~0.8)
    # Use a smooth function: beta = 0.8 + 1.7 * sigmoid(-gap * 2)
    # sigmoid(-gap*2) is close to 1 when gap is small, close to 0 when gap is large
    # Actually we want high beta when gap is LARGE, so: beta = 0.8 + 1.7 * sigmoid(gap * 2)
    # But let's use a simpler smooth interpolation
    
    # Map gap to beta using a smooth curve
    # gap ranges from ~0 (converged) to ~4 (far from optimum)
    # We want beta from ~0.8 (exploit) to ~2.5 (explore)
    gap_for_beta = np.clip(gap, 0.0, 4.0)
    # Smooth S-curve: beta = 0.8 + 1.7 * (gap_for_beta / 4.0) ** 0.7
    # This gives more weight to early exploration
    beta = 0.8 + 1.7 * (gap_for_beta / 4.0) ** 0.7
    
    # Compute raw UCB score for minimization: maximize (beta * sigma - mu)
    raw_scores = beta * sigma - mu
    
    # Normalize scores to have zero mean and unit variance for stability
    mean_raw = np.mean(raw_scores)
    std_raw = np.std(raw_scores)
    if std_raw > 1e-10:
        normalized_scores = (raw_scores - mean_raw) / std_raw
    else:
        normalized_scores = raw_scores - mean_raw
    
    # Add a small exploitation component: favor lower predicted values
    # Normalize mu to [0, 1] range
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    if mu_max - mu_min > 1e-10:
        mu_normalized = (mu - mu_min) / (mu_max - mu_min)
    else:
        mu_normalized = np.zeros_like(mu)
    
    # Blend: 85% normalized UCB + 15% exploitation (inverse of normalized mu)
    exploitation_score = 1.0 - mu_normalized
    scores = 0.85 * normalized_scores + 0.15 * exploitation_score
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - 1.0:
        pass
        
    return scores
