import numpy as np
from scipy.stats import norm
import time

def _sanitize(scores):
    """Ensure scores are finite and of the correct shape."""
    if scores.shape != (2000,):
        scores = np.resize(scores, (2000,))
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit, leave 20s safety margin
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-8)
    mu_safe = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=f_best)
    f_best_safe = np.nan_to_num(f_best, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    # Strategy: Thompson Sampling Proxy
    # We approximate the Thompson Sampling score by ranking candidates based on
    # the standardized improvement z-score: z = (f_best - mu) / sigma.
    # 
    # In standard Thompson Sampling, one would sample f ~ N(mu, sigma) and pick max f.
    # The expected rank or probability of being the maximum is monotonically related
    # to the z-score (especially in the tails). 
    # 
    # Maximizing (f_best - mu) / sigma is equivalent to maximizing the probability 
    # that f(x) < f_best - delta for a fixed delta proportional to sigma, 
    # or more simply, it is a scale-invariant measure of how "impressive" the 
    # improvement is relative to the local uncertainty.
    #
    # This avoids the scale mismatch between EI (objective units) and LCB (objective units)
    # that can occur if f_best is far from 0 or if sigma varies drastically.
    # It also naturally adapts: if sigma is large, a small improvement is less impressive 
    # (lower z), encouraging exploration of high-uncertainty areas only if they have 
    # sufficiently low mean, but preventing pure noise-chasing by requiring the mean 
    # to be low enough relative to the uncertainty.
    
    # Compute the z-score (Thompson Sampling proxy)
    # We use a small epsilon in denominator to avoid division by zero, 
    # though sigma_safe handles the main case.
    z_scores = (f_best_safe - mu_safe) / sigma_safe
    
    # To make the selection more robust and slightly favor exploitation when 
    # uncertainty is very low (high confidence), we can apply a mild non-linearity.
    # However, the directive suggests maximizing the score (f_best - mu) / (sigma + epsilon).
    # Let's stick to the pure proxy as it is a distinct structural change from the 
    # linear combination in the parent.
    
    # One potential issue with pure z-score: if sigma is extremely small everywhere,
    # z-scores become huge and noisy. 
    # If sigma is extremely large, z-scores are small.
    # This is generally desirable for TS.
    
    # Let's verify the direction: 
    # We want to maximize the score. 
    # Higher z means (f_best - mu) is large relative to sigma. 
    # This implies mu is much lower than f_best, and/or sigma is small.
    # This correctly prioritizes points that are likely to be better than f_best.
    
    final_scores = z_scores
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Fallback to EI if we somehow ran out of time (unlikely for this simple calc)
        z_ei = (f_best_safe - mu_safe) / sigma_safe
        ei_scores = (f_best_safe - mu_safe) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
        return _sanitize(ei_scores)
    
    # Ensure finite
    final_scores = _sanitize(final_scores)
    
    return final_scores
