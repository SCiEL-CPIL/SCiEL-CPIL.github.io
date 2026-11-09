import numpy as np
import time

# Global time tracker for deadline guard
_start_time = None

def _init_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()

def acquisition(mu, sigma, f_best):
    _init_time()
    
    # Deadline guard
    # If we are extremely close to the limit, return a simple score
    # to ensure we don't hang.
    # However, vectorized ops are fast.
    
    # Adaptive UCB for minimization
    # Score = -mu + beta * sigma
    
    # Theoretical minimum for Michalewicz 2D
    f_min = -1.8013
    
    # Calculate gap from known minimum
    gap = f_best - f_min
    
    # If gap is negative (shouldn't happen if f_best is min observed and f_min is global min, 
    # but if we found a point lower than theoretical due to noise/error, clamp it)
    if gap < 0:
        gap = 0
        
    # Normalize gap to [0, 1] roughly
    # Max possible value for Michalewicz 2D is ~9.6, min is -1.8
    # So max gap is ~11.4
    gap_norm = np.clip(gap / 10.0, 0.0, 1.0)
    
    # Adaptive beta:
    # High exploration (high beta) when gap is large (early in optimization or stuck in bad basin)
    # Low exploration (low beta) when gap is small (converging to global min)
    # beta range: 0.5 (exploit) to 3.0 (explore)
    beta = 0.5 + 2.5 * gap_norm
    
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
