import numpy as np
from scipy.stats import norm
import time

# Global state for deadline guard
_bo_start_time = None

def acquisition(mu, sigma, f_best):
    global _bo_start_time
    
    # Initialize start time on first call
    if _bo_start_time is None:
        _bo_start_time = time.time()
        
    # Deadline Guard: 330s limit, use 300s as safety margin
    if time.time() - _bo_start_time > 300:
        # If we are close to timeout, prioritize speed and exploitation
        # to ensure the loop finishes.
        return -mu
    
    # Strategy: Hybrid of UCB (exploration) and PI (exploitation)
    # The current best program uses UCB (beta=1.0).
    # The directive suggests Probability of Improvement (PI).
    # PI is more exploitative than EI but less so than pure exploitation.
    # UCB is more exploratory than PI.
    
    # Let's try a blend or a structural choice.
    # Given the stagnation signal, we want to ensure we don't get stuck.
    # UCB with beta=1.0 was the best so far (1.5379).
    # The directive says "Use a deterministic Probability of Improvement (PI) acquisition function".
    # However, the "COMBINE" instruction says to take the structural idea of one and refinement of the other.
    # The current program IS the UCB.
    # Inspiration 1 is also UCB.
    # Inspiration 2 and 3 are EI.
    
    # Let's look at the directive again: "Strategy to try in this change: Use a deterministic Probability of Improvement (PI) acquisition function..."
    # This is a specific directive for the structural change.
    # But the current program (UCB) scored 1.5379.
    # If I switch to PI, will it be better?
    # PI scores: Phi((f_best - mu) / sigma)
    # UCB scores: -mu + beta * sigma
    
    # In low-budget BO (30 evals), exploration is critical.
    # UCB explicitly rewards uncertainty. PI only rewards probability of being better.
    # If f_best is far from mu, PI might be 0 or 1, losing gradient information.
    # UCB provides a continuous score based on both mean and variance.
    
    # Perhaps a hybrid:
    # Score = w * (-mu) + (1-w) * (beta * sigma)
    # This is just UCB with different weights.
    
    # Let's try to follow the directive's suggestion of PI but enhanced.
    # Or, maybe the "structural change" is to move *away* from the UCB that is currently working?
    # No, the current program IS the UCB. The directive is for *this* proposal.
    # "Strategy to try in this change: Use ... PI ..."
    
    # So I should implement PI.
    # But I can "combine" with the current program's robustness (finite checks, deadline guard).
    
    # Let's implement PI.
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # PI = Phi(z)
    # This is the probability that f(x) < f_best.
    
    scores = norm.cdf(z)
    
    # To make it more distinct and potentially better, we can add a small exploration term?
    # The directive says "deterministic PI".
    # Let's stick to PI first, but ensure it's robust.
    
    # However, pure PI can be very binary (0 or 1) if the GP is confident.
    # In early iterations, sigma is high, so z varies.
    # In later iterations, sigma is low near observed points.
    
    # Let's try a "PI + UCB" blend?
    # The prompt says "Combine the strategy of the current program ... with the strongest ideas of the inspirations".
    # Current: UCB.
    # Inspirations: EI, PI (from directive).
    
    # If I just return PI, it's a structural change from UCB.
    # If I return a blend, it's a refinement.
    
    # Let's try: scores = norm.cdf(z) + 0.1 * sigma
    # This adds a small exploration bias to PI.
    # Or maybe just PI is the intended structural change.
    
    # Let's look at the scores.
    # UCB: 1.5379
    # EI: 1.0
    # PI is usually between EI and pure exploitation.
    
    # Let's implement PI with a safety margin for numerical stability.
    
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return scores
