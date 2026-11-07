"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Calculates Upper Confidence Bound (UCB) for each candidate.
    
    Uses an increased exploration constant (kappa=2.0) compared to the previous 
    iteration (kappa=1.0). For a limited budget (approx 20-25 evaluations) on a 
    multimodal function like Hartmann 3D, aggressive exploration is critical to 
    avoid getting trapped in local minima. The global minimum is uniformly 
    distributed in the inner 80% of the domain, so a cost penalty is not 
    applied to avoid biasing the search away from potential minimum locations 
    in the higher-cost (higher coordinate) regions.
    
    The score is defined as:
    Score = -mu + kappa * sigma
    
    Higher scores are preferred.
    """
    # Ensure sigma is positive to avoid division issues or negative exploration
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Increased exploration factor for better global search capability
    kappa = 2.0
    
    # UCB for minimization
    ucb_score = -mu + kappa * sigma_safe
    
    # Ensure finiteness
    ucb_score = np.where(np.isfinite(ucb_score), ucb_score, -np.inf)
    
    return ucb_score
# EVOLVE-BLOCK-END
