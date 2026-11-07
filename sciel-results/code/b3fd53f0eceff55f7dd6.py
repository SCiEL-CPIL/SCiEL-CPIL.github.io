import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division by zero or invalid z
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Cost-aware Lower Confidence Bound (LCB) for minimization
    # Score = -mu + beta * sigma / cost
    # The exploration term (uncertainty) is scaled by 1/cost, penalizing high-cost regions more heavily
    # in the uncertainty bonus, while the mean term (-mu) remains invariant.
    # This prioritizes high-uncertainty, low-cost areas.
    
    # Choose a moderate beta. Since the exploration term is divided by cost (which is > 1),
    # we can use a slightly higher beta than if we were dividing the whole score by cost.
    # Previous UCB/cost used beta=2.0.
    # Here, exploration is beta * sigma / cost.
    # If cost ~ 1.5, exploration is ~ beta * 0.66 * sigma.
    # To get similar magnitude to beta=2 * sigma / 1.5 ~ 1.33 * sigma, beta=2 is reasonable.
    # Let's try beta = 3.0 to ensure exploration is not too suppressed by the cost division.
    beta = 3.0
    
    # Exploration term scaled by inverse cost
    exploration = beta * sigma_safe / cost
    
    # Mean term (invariant to cost)
    exploitation = -mu
    
    scores = exploitation + exploration
    
    # Handle any potential non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
