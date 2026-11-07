"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) for minimization.
    
    Strategy:
    Instead of Expected Improvement (EI), which estimates the probability-weighted 
    improvement, we use UCB. For a minimization problem, we want to select points 
    with low mean prediction (mu) and/or high uncertainty (sigma).
    
    Score = -mu + beta * sigma
    
    A higher score indicates a candidate that is either predicted to have a low 
    objective value (exploitation) or has high predictive uncertainty (exploration).
    This linear trade-off is often more robust in multi-modal landscapes than EI, 
    which can be overly conservative if the GP mean is not well-calibrated in 
    unexplored regions.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.nan_to_num(sigma, nan=1e-9, posinf=1e-9, neginf=1e-9)
    
    # Ensure mu is finite
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Beta controls the exploration-exploitation trade-off.
    # Based on empirical results from similar tasks, beta=1.0 provides a strong 
    # balance for 2D multimodal functions like Branin within ~30 evaluations.
    beta = 1.0
    
    # Calculate UCB score for minimization
    # We want to maximize this score.
    # Low mu -> high score (due to -mu)
    # High sigma -> high score (due to +beta*sigma)
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
