import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
        # Constants
        f_min = 0.397887
        # Estimate of initial best (random points on Branin)
        # Branin range is roughly [0, 300]
        # Initial random points usually yield f_best ~ 50-150
        # Let's assume a scale for the "distance"
        
        # Calculate z
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        
        # Expected Improvement
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        
        # UCB Score (for minimization, we maximize -mu + beta*sigma)
        # We need a beta. Let's start with a standard value.
        # Adaptive beta?
        # If f_best is close to f_min, we want to exploit, so beta should be small?
        # Or we just switch to EI.
        
        # Let's use a blend weight based on f_best
        # distance from optimum
        dist = f_best - f_min
        # Scale distance. If dist > 20, we are "far".
        # Use a sigmoid to map dist to [0, 1]
        # w_explore is high when dist is large.
        # w_explore = 1 / (1 + np.exp(-(dist - 10) / 5))
        # If dist = 10, w=0.5. If dist = 20, w~0.88. If dist = 0, w~0.12.
        
        # However, the prompt mentions "adaptive beta".
        # Let's make beta adaptive too.
        # Beta should be larger when exploring.
        # beta = 2.0 + 2.0 * w_explore ?
        
        # Let's try a simpler structural change first:
        # Use EI but with a "temperature" or penalty?
        # No, the hint is specific about POI/UCB blend.
        
        # Let's try the blend of EI and UCB.
        # To avoid scale issues, we can normalize them?
        # No, keep it simple.
        
        # Adaptive parameters
        # As f_best decreases, we shift to EI.
        # Let's define a threshold.
        # If f_best < 1.0, we are very close. Use mostly EI.
        # If f_best > 10.0, we are far. Use mostly UCB.
        
        # Linear interpolation of weight
        # w_ei = 1.0 if f_best < 1.0
        # w_ei = 0.0 if f_best > 20.0
        # w_ei = (20.0 - f_best) / (20.0 - 1.0)
        
        w_ei = np.clip((20.0 - f_best) / 19.0, 0.0, 1.0)
        w_ucb = 1.0 - w_ei
        
        # Beta for UCB
        # When exploring (w_ucb high), we want high beta.
        # When exploiting (w_ei high), beta doesn't matter as much, but let's keep it moderate.
        beta = 1.0 + 2.0 * w_ucb  # Beta ranges from 1.0 to 3.0
        
        ucb_score = -mu + beta * sigma
        
        # Combine
        # EI is positive. UCB can be negative.
        # If UCB is negative, it's a bad candidate for exploration?
        # Actually, for minimization, we want low values.
        # If mu is high, -mu is very negative.
        # EI is 0 if mu >= f_best.
        # So EI is "safe". UCB is "risky".
        
        # If we blend, we might get negative scores for bad candidates.
        # The task requires "finite scores". It doesn't say they must be positive.
        # "Higher meaning evaluate next".
        
        score = w_ei * ei + w_ucb * ucb_score
        
        # Ensure finite
        score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return score
