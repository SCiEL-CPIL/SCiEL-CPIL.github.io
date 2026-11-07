"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Lower Confidence Bound (LCB) for minimization.
    # Score is higher when the predicted lower bound (mu - kappa*sigma) is much lower than f_best,
    # and when the cost is low.
    # This linear combination is robust for multimodal functions like Griewank and avoids
    # the numerical issues of EI's CDF/PDF in the tails.
    
    kappa = 2.0
    # Ensure cost is at least 1 to avoid division by zero or extreme values, 
    # though task says cost is 1 + mean(x_scaled) so it's >= 1.
    safe_cost = np.maximum(cost, 1e-6)
    
    # LCB for minimization: we want to minimize f(x).
    # The lower bound on the value is mu - kappa * sigma.
    # We want this to be as small as possible.
    # The "improvement" over f_best is f_best - (mu - kappa * sigma).
    # We divide by cost to penalize expensive evaluations.
    
    scores = (f_best - mu + kappa * sigma) / safe_cost
    
    # Ensure finite outputs
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
