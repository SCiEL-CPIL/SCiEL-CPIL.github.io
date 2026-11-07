"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track evaluation count for adaptive exploration
_eval_count = 0

def acquisition(mu, sigma, f_best):
    global _eval_count
    
    # Deadline guard
    t0 = time.time()
    deadline = t0 + 280.0  # Safety margin before 300s limit
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))):
        return np.arange(len(mu), dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Increment evaluation count
    _eval_count += 1
    
    # Adaptive exploration parameter (beta)
    # Start with higher exploration to escape local minima in 6D, 
    # decay it slowly as we gather more information.
    # Hartmann 6D is multimodal. EI (current) is often too greedy.
    # LCB with adaptive beta helps balance exploration/exploitation.
    
    # Heuristic for beta: 
    # beta ~ log(t) / t is common, but for small t (max 30), 
    # a simple decay or constant might be better.
    # Let's try a schedule that starts high (2.0) and decays to ~0.5
    # This encourages visiting new areas in the first 10-15 steps.
    # beta = 2.0 * (1.0 / (1.0 + 0.1 * _eval_count))
    
    # Alternatively, a constant beta of 1.5 or 2.0 is often better than EI for multimodal functions
    # in the early stages. Let's use a mild decay.
    beta = 1.5 + 0.5 * np.exp(-0.1 * _eval_count)
    
    # Lower Confidence Bound for minimization
    # We want to minimize f(x).
    # LCB = mu - beta * sigma
    # Higher LCB score means "more promising to be low".
    # So we return LCB. The evaluator picks the HIGHEST score.
    # Wait, if LCB = mu - beta*sigma, a very low mu and high sigma gives a very negative number.
    # We want the point with the *lowest* predicted value with uncertainty.
    # So we want to maximize (mu - beta * sigma)? No.
    # If we pick the argmax of (mu - beta * sigma), we pick points that have low mean and high uncertainty.
    # Let's check signs.
    # If mu is -3.0 (good) and sigma is 0.1, LCB = -3.15.
    # If mu is 0.0 (bad) and sigma is 1.0, LCB = -1.5.
    # -3.15 < -1.5. So argmin would pick the good point.
    # But the evaluator picks the HIGHEST score.
    # So we should return -LCB = -mu + beta * sigma?
    # Or simply return (beta * sigma - mu)?
    # Let's define score = beta * sigma - mu.
    # High sigma increases score (exploration).
    # Low mu increases score (exploitation).
    # This is equivalent to maximizing the upper confidence bound for minimization.
    
    scores = beta * sigma - mu
    
    # Check for non-finite values
    if not np.all(np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
        
    # Check deadline
    if time.time() > deadline:
        # Return best found so far? No, just return the computed scores.
        # The computation is O(N) and very fast.
        pass

    return scores
# EVOLVE-BLOCK-END
