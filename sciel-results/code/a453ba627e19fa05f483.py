import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
        # Known minimum
        f_min = -2.53218
        
        # Estimate progress
        # Assume f_best starts near 0 and goes to f_min
        # If f_best > 0, it's very early.
        # Let's cap the progress calculation to [0, 1]
        # A simple heuristic: 
        # If f_best is close to f_min, reduce exploration.
        
        # Heuristic for beta:
        # Start with a high exploration factor, decrease as we get closer to the optimum.
        # The knowledge suggests 2.5 is good.
        # Let's try: beta = 2.5 + 1.0 * max(0, (f_best - f_min) / 2.0)
        # If f_best is far from f_min (large gap), beta is higher.
        # If f_best is close to f_min (small gap), beta is lower (closer to 2.5? or lower?).
        
        # Actually, if the gap is large, we are in unexplored territory, so high beta is good.
        # If the gap is small, we are polishing, so lower beta is good.
        
        gap = f_best - f_min
        # Scale gap. Typical RBF values might be in [-3, 3].
        # Let's assume a max gap of 5.0 for scaling.
        scale = 5.0
        gap_norm = min(1.0, max(0.0, gap / scale))
        
        # Beta starts at 3.0 (high exploration) and drops to 1.5 (lower exploration)
        beta = 1.5 + 1.5 * gap_norm
        
        # Calculate UCB score for minimization
        # We want to maximize the score to pick the next point.
        # The "Lower Confidence Bound" of the function is mu - beta * sigma.
        # To pick the best candidate (lowest function value), we want the candidate with the lowest LCB.
        # But the prompt requires HIGHER scores to mean "evaluate next".
        # So we return -LCB = beta * sigma - mu.
        
        scores = beta * sigma - mu
        
        # Ensure finite
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return scores
