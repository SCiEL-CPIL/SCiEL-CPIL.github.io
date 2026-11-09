"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s margin before 330s kill

def _check_deadline():
    return time.time() > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a model-and-optimize framework.
    
    Strategy:
    1. Compute Expected Improvement (EI) for each candidate.
    2. Define a family of cost-aware acquisition functions parameterized by gamma:
       score(x) = EI(x) / (cost(x) ** gamma)
    3. Select gamma by maximizing the EI of the selected candidate (argmax score).
       This ensures we pick the gamma that leads to the highest potential improvement,
       implicitly balancing exploration/exploitation and cost.
    """
    if _check_deadline():
        # Fallback: standard EI / cost
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        scores = ei / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # 1. Calculate Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # 2. Grid search over gamma
    # Gamma controls the cost penalty. 
    # Lower gamma -> less penalty on cost.
    # Higher gamma -> more penalty on cost.
    # We test a range that covers standard cost-aware EI (gamma=1) and variants.
    gammas = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    
    best_gamma = 1.0
    best_ei_selected = -np.inf
    best_scores = None
    
    # Precompute log costs for efficiency? No, cost**gamma is fast enough for 2000 points.
    # But we can precompute log(cost) and use exp(gamma * log(cost))
    log_cost = np.log(cost_safe + 1e-6)
    
    for gamma in gammas:
        if _check_deadline():
            break
            
        # Score = EI / (cost ** gamma)
        # = EI * exp(-gamma * log(cost))
        scores = ei * np.exp(-gamma * log_cost)
        
        # Handle non-finite
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        
        # Find best candidate
        idx_max = np.argmax(scores)
        
        # The "quality" of this gamma is the EI of the selected point.
        # We want to maximize the EI of the point we would actually evaluate.
        ei_selected = ei[idx_max]
        
        if ei_selected > best_ei_selected:
            best_ei_selected = ei_selected
            best_gamma = gamma
            best_scores = scores.copy()
            
    if best_scores is None:
        # Fallback if loop didn't run
        best_scores = ei / cost_safe
        
    # Ensure finite output
    final_scores = np.where(np.isfinite(best_scores), best_scores, 0.0)
    
    return final_scores
# EVOLVE-BLOCK-END
