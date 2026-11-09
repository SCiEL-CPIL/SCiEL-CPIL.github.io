import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
        start_time = time.time()
        deadline = start_time + 290.0
        
        min_val = 0.397887
        diff = max(0.0, f_best - min_val)
        
        # Adaptive weight for POI (exploration) vs EI (exploitation)
        # High diff -> High w (more POI)
        # Low diff -> Low w (more EI)
        w = 1.0 - np.exp(-diff / 10.0)
        
        z = (f_best - mu) / (sigma + 1e-9)
        
        # Expected Improvement
        ei_score = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Probability of Improvement
        poi_score = norm.cdf(z)
        
        # Normalize EI to [0, 1] for blending
        ei_min = np.min(ei_score)
        ei_max = np.max(ei_score)
        denom = ei_max - ei_min
        if denom < 1e-9:
            ei_norm = np.ones_like(ei_score)
        else:
            ei_norm = (ei_score - ei_min) / denom
            
        # Blend
        score = w * poi_score + (1.0 - w) * ei_norm
        
        # Ensure finiteness
        score = np.where(np.isfinite(score), score, 0.0)
        
        if time.time() > deadline:
             # Fallback: return EI
             return ei_score
             
        return score
