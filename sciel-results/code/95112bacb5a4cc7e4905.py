import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (griewank 2d).
    
    Uses a dynamic cost-aware Expected Improvement (EI) where the exploration
    coefficient (kappa) is optimized to maximize the expected acquisition score
    per unit cost. This adapts the exploration/exploitation balance to the
    current GP posterior and cost landscape.
    
    Parameters:
    mu: array of shape (2000,) - Mean of the GP posterior
    sigma: array of shape (2000,) - Standard deviation of the GP posterior
    f_best: float - Best observed value so far
    cost: array of shape (2000,) - Cost of evaluating each candidate
    
    Returns:
    scores: array of shape (2000,) - Acquisition scores (higher is better)
    """
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-scores for EI
    z = (f_best - mu) / sigma_safe
    
    # Base EI components
    # EI_kappa = (f_best - mu - kappa*sigma) * Phi(z - kappa) + sigma * phi(z - kappa)
    # For cost-aware, we want to maximize E[Improvement] / Cost
    # Approximate by maximizing the expected score per unit cost
    
    # Strategy: Optimize kappa to balance exploration and cost
    # We use a grid search over kappa to find the one that maximizes
    # the average (EI_kappa / cost) across candidates, but weighted by
    # the probability of being selected (softmax of scores)
    
    # For efficiency, we'll use a small grid of kappa values
    # and select the best one based on a heuristic score
    
    # Heuristic: The optimal kappa should increase as uncertainty (sigma) increases
    # relative to the potential improvement (f_best - mu)
    
    # Compute a baseline score using a standard EI with a moderate kappa
    # Then adjust based on cost distribution
    
    # Let's try a few candidate kappa values
    # kappa = 0 is pure exploitation (EI)
    # kappa > 0 adds exploration (UCB-like)
    
    # We'll evaluate a few kappa values and pick the best
    # based on the expected improvement per unit cost
    
    best_scores = None
    best_avg_gain_per_cost = -np.inf
    
    # Grid of kappa values to try
    # We want to cover a range from exploitation to exploration
    kappa_candidates = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    
    for kappa in kappa_candidates:
        if time.time() > deadline:
            break
        
        # Compute EI with this kappa
        # EI_kappa = (f_best - mu - kappa*sigma) * Phi(z - kappa) + sigma * phi(z - kappa)
        z_kappa = z - kappa
        
        # Clip z_kappa for numerical stability
        z_kappa_clipped = np.clip(z_kappa, -10, 10)
        
        ei_kappa = (f_best - mu - kappa * sigma_safe) * norm.cdf(z_kappa_clipped) + sigma_safe * norm.pdf(z_kappa_clipped)
        
        # Ensure non-negative
        ei_kappa = np.maximum(ei_kappa, 0.0)
        
        # Cost-aware score
        scores_kappa = ei_kappa / cost_safe
        
        # Heuristic for selecting kappa: 
        # We want to maximize the expected gain per cost, but also ensure diversity
        # Use the top-1% average score as a proxy for the expected next evaluation
        n_top = max(1, len(scores_kappa) // 100)
        top_scores = np.sort(scores_kappa)[-n_top:]
        avg_top = np.mean(top_scores)
        
        # Also consider the variance of scores to encourage diversity
        # A higher variance in scores might indicate more exploration opportunities
        score_var = np.var(scores_kappa)
        
        # Combined heuristic: favor higher average top scores and moderate variance
        heuristic_score = avg_top + 0.01 * np.sqrt(score_var)
        
        if heuristic_score > best_avg_gain_per_cost:
            best_avg_gain_per_cost = heuristic_score
            best_scores = scores_kappa
    
    # If no valid scores were computed, fall back to simple EI
    if best_scores is None:
        # Simple EI with kappa=0
        z_clipped = np.clip(z, -10, 10)
        ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
        ei = np.maximum(ei, 0.0)
        best_scores = ei / cost_safe
    
    # Ensure all values are finite
    best_scores = np.nan_to_num(best_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return best_scores
