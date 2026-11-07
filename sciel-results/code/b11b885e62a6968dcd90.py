import numpy as np

def acquisition(mu, sigma, f_best):
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        
        # Clip sigma to avoid division by zero or numerical issues in downstream processes
        # Although we don't divide by sigma here, it's good practice.
        sigma = np.maximum(sigma, 1e-12)
        
        fb = float(f_best)
        
        # Ensure fb is non-negative for the heuristic, though sphere is non-negative
        # If fb is negative (shouldn't happen), clamp it.
        if fb < 0:
            fb = 0.0
            
        # Dynamic Beta
        # Starts high (exploration) when fb is large, drops to low (exploitation) when fb is small.
        beta = 1.0 + 0.2 * fb
        
        # UCB Score for minimization: Maximize (beta * sigma - mu)
        score = beta * sigma - mu
        
        # Replace non-finite values with a very low finite number
        score = np.where(np.isfinite(score), score, -1e18)
        
        return score
