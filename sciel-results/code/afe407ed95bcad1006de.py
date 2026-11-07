"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track time for safety
_START_TIME = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using adaptive Lower Confidence Bound (LCB).
    
    LCB = mu - beta * sigma
    
    We adapt beta based on f_best. 
    - When f_best is large (early), we use a high beta to explore.
    - When f_best is small (late), we use a low beta to exploit.
    
    This helps escape local plateaus or flat regions in the early stages 
    of 1D sphere minimization where EI might get stuck if the GP is 
    slightly miscalibrated or if the initial points were unlucky.
    """
    # Time guard
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        # Return a safe, neutral score if we are running out of time
        # This shouldn't happen in normal operation as acquisition is fast
        return np.zeros_like(mu)
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Adaptive beta
    # f_best for sphere 1d on [-5,5] starts around 25 and goes to 0.
    # We want beta to be large when f_best is large.
    # A simple mapping: beta = 1.0 + 10.0 / (1.0 + f_best)
    # If f_best = 25, beta ~ 1.4
    # If f_best = 1, beta ~ 6.0
    # If f_best = 0.1, beta ~ 100.0 (Wait, this is backwards for LCB exploration?)
    
    # Let's rethink the adaptation.
    # In early stages, we want to explore. LCB explores by subtracting beta*sigma.
    # If beta is large, we explore more.
    # So we want beta to be large when f_best is large.
    # However, standard LCB uses a constant beta or one that decreases.
    # For minimization, LCB = mu - beta * sigma.
    # If we want to explore when f_best is bad (large), we should have a large beta.
    # If we want to exploit when f_best is good (small), we should have a small beta.
    
    # Let's try: beta = 0.5 + 2.0 * (f_best / (f_best + 1.0))
    # If f_best = 25, beta = 0.5 + 2.0 * (25/26) ~ 2.4
    # If f_best = 1, beta = 0.5 + 2.0 * (1/2) = 1.5
    # If f_best = 0.01, beta = 0.5 + 2.0 * (0.01/1.01) ~ 0.52
    
    # Actually, a simpler and more robust strategy for 1D sphere is to just use a 
    # moderate constant beta that is tuned for the scale. 
    # But the prompt asks for a DIFFERENT strategy. 
    # Let's use a "Probability of Improvement" (PI) hybrid or just a well-tuned LCB.
    # The current program is EI. LCB is a distinct family.
    
    # Let's use a beta that scales with the relative uncertainty.
    # But we don't have the global scale of sigma easily without knowing the step.
    # However, f_best is a good proxy.
    
    # Let's try: beta = 1.0 + 5.0 * np.exp(-f_best)
    # If f_best is large (e.g., 25), exp(-25) is ~0, so beta ~ 1.0.
    # If f_best is small (e.g., 0.1), exp(-0.1) ~ 0.9, so beta ~ 5.5.
    # This is the opposite of what we want! We want to explore MORE when f_best is bad.
    # So we want beta to be LARGE when f_best is LARGE.
    
    # Correct mapping:
    # beta = 1.0 + 5.0 * (1.0 - np.exp(-f_best))
    # If f_best = 25, beta ~ 6.0 (High exploration)
    # If f_best = 1, beta ~ 1.0 + 5.0 * (1 - 0.37) = 1.0 + 3.15 = 4.15
    # If f_best = 0.1, beta ~ 1.0 + 5.0 * (1 - 0.90) = 1.0 + 0.5 = 1.5
    # If f_best = 0, beta = 1.0
    
    # This seems reasonable. High exploration early, lower exploration later.
    
    beta = 1.0 + 5.0 * (1.0 - np.exp(-f_best))
    
    # Calculate LCB
    lcb = mu - beta * sigma_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(lcb), lcb, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
