import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Pure Probability of Improvement (PoI) for minimization.
    
    Strategy:
    Uses the Probability of Improvement (PoI) as the sole acquisition metric.
    PoI estimates the probability that a candidate will have a lower objective
    value than the current best (f_best): P(f(x) < f_best) = Phi((f_best - mu) / sigma).
    
    This decouples exploration from the magnitude of potential improvement,
    favoring broad coverage of the multi-modal Branin landscape over the
    exploitation bias seen in Expected Improvement.
    
    Score = Phi((f_best - mu) / sigma)
    """
    start_time = time.time()
    DEADLINE = 325.0  # Safety margin before 330s kill

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.nan_to_num(sigma, nan=1e-9, posinf=1e-9, neginf=1e-9)
    
    # Ensure mu is finite
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)

    # Calculate z-scores for Probability of Improvement
    # We want P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    
    # Compute CDF of standard normal
    from scipy.special import ndtr
    poi_scores = ndtr(z)
    
    # Ensure poi_scores are finite
    poi_scores = np.nan_to_num(poi_scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Return best effort (though this shouldn't happen for such simple computation)
        return poi_scores
        
    return poi_scores
