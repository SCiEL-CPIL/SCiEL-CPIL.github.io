import numpy as np
from scipy.stats import norm
import time

# Global state to track progress across calls within a seed
_step_count = 0
_start_time = None

def _expected_improvement(mu, sigma, f_best, xi=0.01):
    """Standard Expected Improvement for minimization."""
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu - xi) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)

def _ucb(mu, sigma, kappa):
    """Lower Confidence Bound for minimization (we maximize the negative)."""
    return -mu + kappa * sigma

def _probability_of_improvement(mu, sigma, f_best):
    """Probability of improvement: P(f(x) < f_best)"""
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    return norm.cdf(z)

def _find_nondominated_indices(ei, ucb, pi):
    """
    Find indices of non-dominated points in the 3D space of (EI, UCB, PI).
    A point is non-dominated if no other point is better or equal in all 
    three metrics and strictly better in at least one.
    """
    n = len(ei)
    nondominated = np.ones(n, dtype=bool)
    
    # Vectorized dominance check is O(N^2) which is 4M ops for N=2000. 
    # 2000*2000 = 4,000,000. This is fast in NumPy.
    for i in range(n):
        if not nondominated[i]:
            continue
        
        # Check if any other point j dominates i
        # j dominates i if:
        # ei[j] >= ei[i] and ucb[j] >= ucb[i] and pi[j] >= pi[i]
        # and at least one is strictly greater.
        
        # To be efficient, we can check against all j at once?
        # Let's just do a simple loop or vectorized comparison.
        # Since N=2000, a simple loop over j for each i is 4M iterations. 
        # In Python, 4M iterations might be ~0.1-0.5s. 
        # We have 300s total. 25 steps. ~12s per step. 0.5s is fine.
        
        # However, a fully vectorized approach is faster.
        # Let's try vectorized:
        # mask_j = (ei >= ei[i]) & (ucb >= ucb[i]) & (pi >= pi[i])
        # strict_mask = (ei > ei[i]) | (ucb > ucb[i]) | (pi > pi[i])
        # if np.any(mask_j & strict_mask & (np.arange(n) != i)):
        #     nondominated[i] = False
        #     continue
        
        # Optimization: If we process in order, we can't easily vectorize the "j != i" part without masking.
        # Let's stick to the vectorized check for each i.
        
        mask_j = (ei >= ei[i]) & (ucb >= ucb[i]) & (pi >= pi[i])
        strict_mask = (ei > ei[i]) | (ucb > ucb[i]) | (pi > pi[i])
        
        # Exclude i itself
        mask_j[i] = False
        
        if np.any(mask_j & strict_mask):
            nondominated[i] = False
            
    return np.where(nondominated)[0]

def acquisition(mu, sigma, f_best):
    global _step_count, _start_time
    
    # Initialize time tracking on first call
    if _start_time is None:
        _start_time = time.time()
        _step_count = 0
    
    elapsed = time.time() - _start_time
    _step_count += 1
    
    # Safety margin: return UCB if we're getting close to deadline
    if elapsed > 290:
        scores = _ucb(mu, sigma, 2.0)
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores

    # Compute component scores
    ei_scores = _expected_improvement(mu, sigma, f_best, xi=0.01)
    ucb_scores = _ucb(mu, sigma, kappa=2.0)
    pi_scores = _probability_of_improvement(mu, sigma, f_best)
    
    # Ensure finite
    ei_scores = np.nan_to_num(ei_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    ucb_scores = np.nan_to_num(ucb_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    pi_scores = np.nan_to_num(pi_scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Find non-dominated points
    # If the computation is too slow or fails, fallback to EI
    try:
        nondom_indices = _find_nondominated_indices(ei_scores, ucb_scores, pi_scores)
    except Exception:
        nondom_indices = np.array([np.argmax(ei_scores)])
        
    # Initialize scores to 0 (or -inf)
    scores = np.zeros_like(ei_scores)
    
    # If no non-dominated points found (shouldn't happen, but safe), fallback
    if len(nondom_indices) == 0:
        scores[np.argmax(ei_scores)] = ei_scores[np.argmax(ei_scores)]
    else:
        # Assign EI scores to non-dominated points.
        # This selects the point with the highest EI among the Pareto-optimal set.
        scores[nondom_indices] = ei_scores[nondom_indices]
        
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    return scores
