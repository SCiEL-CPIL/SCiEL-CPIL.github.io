"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling heuristic for minimization.
    
    mu: (N,) posterior mean
    sigma: (N,) posterior std
    f_best: best observed value so far
    
    Returns: (N,) scores, higher is better.
    """
    start_time = time.time()
    
    # Clip sigma to avoid division by zero and ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling heuristic:
    # Score = -mu + beta * sigma
    # We want higher scores to be selected.
    # -mu favors low mean (good for minimization).
    # beta * sigma favors high uncertainty (exploration).
    
    # Dynamic beta:
    # The directive suggests: "exploration weight is scaled by the current best value 
    # to aggressively target high-variance regions early and sharply focus on the 
    # global minimum as f_best approaches zero."
    
    # Ackley 2D has a minimum of 0.
    # Early on, f_best is high (e.g., > 1). We want high beta.
    # As f_best -> 0, we want beta -> 0 (exploitation).
    
    # Let's define beta such that it is large when f_best is large and small when f_best is small.
    # A simple scaling: beta = k * f_best?
    # If f_best is 5, beta is 5k. If f_best is 0.01, beta is 0.01k.
    # However, the scale of sigma and mu matters.
    # In standardized GP, mu and sigma are in objective units.
    # Ackley values range from 0 to ~10-20 depending on location.
    
    # Let's try a dynamic beta that transitions from exploration to exploitation.
    # A common approach is beta = sqrt(log(t)) or similar, but we don't have t directly, 
    # though we can infer progress from f_best.
    
    # The directive specifically says: "scaled by the current best value".
    # Let's use beta = f_best.
    # If f_best is large, exploration is high.
    # If f_best is small, exploration is low.
    
    # However, we need to ensure the units match.
    # Score = -mu + f_best * sigma
    # Let's check the magnitude.
    # If mu ~ 1, sigma ~ 0.5, f_best ~ 2.
    # Score ~ -1 + 1 = 0.
    # If mu ~ 0.1, sigma ~ 0.1, f_best ~ 0.1.
    # Score ~ -0.1 + 0.01 = -0.09.
    # If mu ~ 0.1, sigma ~ 0.1, f_best ~ 2.
    # Score ~ -0.1 + 0.2 = 0.1.
    # This seems reasonable.
    
    # To be safe and ensure we don't over-explore when f_best is already very good,
    # we can cap beta or use a saturating function.
    # But the directive is specific: "scaled by the current best value".
    
    # Let's refine the scaling. If f_best is 0, beta should be 0.
    # If f_best is large, beta should be large.
    
    # One issue: if f_best is very small (e.g., 1e-3), beta is tiny.
    # This forces exploitation.
    
    # Let's add a small floor to beta to avoid completely stopping exploration?
    # "sharply focus on the global minimum as f_best approaches zero" implies beta should go to 0.
    
    beta = f_best
    
    # Calculate scores
    scores = -mu + beta * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 290:
         # This shouldn't happen for O(N) ops, but just in case
         pass
        
    return scores
# EVOLVE-BLOCK-END
