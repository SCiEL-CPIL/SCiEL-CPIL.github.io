import numpy as np

# Global state for tracking evaluation steps and detecting new seeds
_prev_f_best = float('inf')
_step = 0
_TOTAL_STEPS = 30  # Max evaluations per seed

def acquisition(mu, sigma, f_best):
    global _prev_f_best, _step
    
    # Detect new seed: if f_best is significantly higher than previous, 
    # it's likely a new run (since f_best should decrease or stay same within a run)
    if f_best > _prev_f_best + 1e-9:
        _step = 0
    else:
        _step += 1
        
    _prev_f_best = f_best
    
    # Ensure step is within bounds
    if _step > _TOTAL_STEPS:
        _step = _TOTAL_STEPS
        
    # Calculate beta with quadratic decay
    # Start with high exploration, end with low exploration
    # beta(t) = beta_max * (1 - t/T)^2
    # Let's use beta_max = 2.5
    beta = 2.5 * (1 - _step / _TOTAL_STEPS) ** 2
    
    # LCB: minimize mu - beta * sigma
    # We return score where higher is better.
    # So we return -(mu - beta * sigma) = -mu + beta * sigma
    # Wait. 
    # If we minimize f, we want the candidate with the lowest expected f.
    # The "score" returned is used by the evaluator to pick the MAX score.
    # So we need to map "goodness" to "high score".
    # Goodness for minimization: Low mu, High sigma (exploration).
    # LCB value = mu - beta * sigma. We want to MINIMIZE this LCB value?
    # No, LCB is an estimate of the lower bound. We want to find the point with the lowest lower bound.
    # So we want to MINIMIZE (mu - beta * sigma).
    # But the evaluator picks the MAX score.
    # So we should return: Score = -(mu - beta * sigma) = -mu + beta * sigma.
    
    # Let's verify.
    # Candidate A: mu=0, sigma=0. Score = 0.
    # Candidate B: mu=1, sigma=2, beta=1. LCB = 1 - 2 = -1. Score = -(-1) = 1.
    # Candidate B is preferred because its lower bound is lower.
    # This seems correct.
    
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
