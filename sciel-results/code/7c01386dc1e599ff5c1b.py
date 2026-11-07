import numpy as np

def acquisition(mu, sigma, f_best):
    try:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        
        # Lower Confidence Bound (LCB) for minimization
        # We want to maximize the score, so we use -mu + beta * sigma
        # beta controls the exploration-exploitation trade-off
        # A value of 2.0 is a standard heuristic for low-noise GPs
        beta = 2.0
        
        scores = -mu + beta * sigma
        
        # Ensure scores are finite
        if not np.all(np.isfinite(scores)):
            scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
            
        return scores
    except Exception:
        # Fallback: return zeros or random if something goes wrong
        return np.zeros_like(mu)
