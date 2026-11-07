import numpy as np
import time

# State variables for adaptive exploration
# These persist across calls within the same evaluator run (same seed)
_bo_state = {
    'call_count': 0,
    'initial_f_best': None,
    'best_f_best': None,
    'base_beta': 0.5,       # Initial exploration weight
    'min_beta': 0.05,       # Minimum exploration weight
    'decay_rate': 0.95,     # Decay factor per step
    'progress_boost': 0.0   # Temporary boost if significant improvement found
}

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Adaptive Lower Confidence Bound (LCB).
    
    Strategy:
    1. Compute LCB score: -mu - beta * sigma.
       Higher scores are preferred (we maximize this, equivalent to minimizing mu + beta*sigma).
    2. The exploration weight `beta` is adaptive:
       - Starts at a moderate value (0.5) to explore the wide, multimodal Weierstrass function.
       - Decays over time (call_count) to shift towards exploitation.
       - Is boosted if a significant new minimum is found (f_best drops), allowing 
         the algorithm to "reset" its exploration focus around the new promising region 
         without getting stuck in a local basin.
    
    This differs from the current Thompson Sampling (stochastic) by being deterministic 
    and structured, and from static UCB by dynamically adjusting the exploration/exploitation 
    trade-off based on optimization progress.
    """
    
    # Update state
    _bo_state['call_count'] += 1
    
    if _bo_state['initial_f_best'] is None:
        _bo_state['initial_f_best'] = f_best
    if _bo_state['best_f_best'] is None:
        _bo_state['best_f_best'] = f_best
    
    # Check for significant improvement to boost exploration
    # Weierstrass 1D has a minimum of 0. f_best is the lowest observed.
    # If f_best drops by more than a small epsilon, we consider it a new region found.
    improvement = _bo_state['best_f_best'] - f_best
    if improvement > 0.01: # Threshold for "significant" improvement
        _bo_state['best_f_best'] = f_best
        # Boost beta temporarily to explore around the new minimum
        _bo_state['progress_boost'] = 0.3 
    else:
        # Decay the boost
        _bo_state['progress_boost'] *= 0.8
        
    # Calculate current beta
    # Base decay: beta decreases as we take more steps
    # The floor is min_beta to ensure some exploration always remains
    current_beta = max(_bo_state['min_beta'], _bo_state['base_beta'] * (_bo_state['decay_rate'] ** _bo_state['call_count']))
    
    # Add the progress boost
    current_beta += _bo_state['progress_boost']
    
    # Numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # LCB Score: We want to minimize f(x).
    # GP posterior: f(x) ~ N(mu, sigma^2)
    # Lower Confidence Bound for minimization: LCB(x) = mu(x) - beta * sigma(x)
    # We want to pick the x with the lowest LCB.
    # The interface requires returning scores where HIGHER means evaluate next.
    # So we return -LCB(x) = -(mu(x) - beta * sigma(x)) = -mu(x) + beta * sigma(x)
    
    scores = -mu + current_beta * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
