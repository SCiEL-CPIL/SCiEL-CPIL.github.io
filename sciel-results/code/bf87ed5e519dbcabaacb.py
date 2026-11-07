import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
        # Adaptive Beta for LCB
        # f_best is the current best value.
        # As f_best decreases, we want to exploit more (lower beta).
        # Ackley min is 0.
        # If f_best is 2.0, exp(-2) is 0.13, beta ~ 2.26
        # If f_best is 0.1, exp(-0.1) is 0.9, beta ~ 3.8? No, that's wrong direction.
        # We want HIGH beta when f_best is BAD (high).
        # We want LOW beta when f_best is GOOD (low).
        
        # So beta = base + scale * exp(-f_best) is WRONG.
        # exp(-f_best) is large when f_best is small.
        # So beta would be large when f_best is small.
        # We want the opposite.
        
        # Use beta = base + scale * exp(-f_best) is for when we want to explore MORE when we are close? No.
        # Standard: Explore when uncertain.
        # If f_best is high, we are far from optimum. We should explore.
        # If f_best is low, we are close. We should exploit.
        
        # So beta should be HIGH when f_best is HIGH.
        # beta = base + scale * f_best?
        # Or beta = base + scale * exp(f_best)?
        
        # Let's use a simple linear decay or exponential.
        # beta = 1.0 + 2.0 * exp(-f_best) -> No.
        
        # Let's try: beta = 4.0 * exp(-0.5 * f_best)
        # If f_best = 2.0, beta = 4 * exp(-1) = 1.47
        # If f_best = 0.1, beta = 4 * exp(-0.05) = 3.8
        # This is INVERTED. We want high beta when f_best is high.
        
        # Correct: beta = 4.0 * exp(0.5 * f_best) ?
        # If f_best = 2.0, beta = 4 * exp(1) = 10.8 (Too high?)
        # If f_best = 0.1, beta = 4 * exp(0.05) = 4.2
        
        # Let's cap it.
        # beta = min(4.0, 1.0 + f_best)
        # If f_best = 2.0, beta = 3.0
        # If f_best = 0.1, beta = 1.1
        
        beta = 1.0 + np.exp(-f_best) # No, exp(-f_best) is high when f_best is low.
        
        # Let's just use a fixed beta = 3.0. It's safer.
        beta = 3.0
        
        scores = (beta * sigma - mu) / cost
        return scores
