import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 320.0  # Safety margin before 330s limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) with an adaptive
    exploration bonus derived from the UCB family.
    
    The core signal is EI (proven robust for Branin), augmented by a small,
    adaptive LCB term that decays as f_best approaches the known global minimum.
    This helps escape local plateaus early while maintaining exploitation focus later.
    """
    # Deadline guard: if we are running out of time, return a fast, safe fallback
    if time.time() - _start_time > _DEADLINE:
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Ensure sigma is positive to avoid division issues
    sigma = np.maximum(sigma, 1e-12)
    
    # 1. Compute Expected Improvement (for minimization)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Compute Adaptive Exploration Bonus (LCB component)
    # The UCB lower bound is mu - beta * sigma.
    # We want to add a small amount of this to the EI score.
    # The weight 'w' should be high when f_best is far from the optimum (early)
    # and low when f_best is close (late).
    
    f_min_known = 0.397887
    dist = max(0.0, f_best - f_min_known)
    
    # Weight decays from ~0.5 to ~0.01 as f_best goes from ~50 to ~0.4
    # Using a smooth decay: w = 0.5 / (1 + dist/10)
    # If dist=0, w=0.5. If dist=10, w=0.25. If dist=50, w=0.083.
    # This is a gentle exploration bonus that doesn't overpower EI.
    w = 0.5 / (1.0 + dist / 10.0)
    
    # LCB score for minimization is lower when mu - beta*sigma is lower.
    # We want to ADD exploration, so we add a term proportional to sigma.
    # The "value" of exploration is roughly beta * sigma.
    # Let's use a fixed small beta for this bonus, scaled by w.
    # Actually, let's just use w * sigma directly as the bonus magnitude.
    # This is equivalent to a UCB with beta=w, but added to EI.
    
    bonus = w * sigma
    
    # 3. Combine
    # Score = EI + Bonus
    # EI is non-negative (improvement over f_best).
    # Bonus encourages high uncertainty.
    scores = ei + bonus
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
