"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure inputs are valid
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety checks
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma)) and np.all(np.isfinite(cost))):
        return np.zeros(2000)
        
    # Protect against zero/negative sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Cost is the total accumulated cost so far? Or the cost of evaluating this point?
    # The prompt says: "Evaluating x costs 1 + mean(x scaled to [0, 1])."
    # "cost" input shape is (2000,). In the loop description: "evaluator scores 2000 new uniform candidates with acquisition".
    # Usually, 'cost' in such signatures refers to the cost of *evaluating* each of the 2000 candidates, 
    # OR the current total budget spent. 
    # Given the signature `acquisition(mu, sigma, f_best, cost)` and `cost` has shape (2000,), 
    # it is highly likely `cost` is the evaluation cost for each of the 2000 candidates.
    # Let's assume `cost[i]` is the cost to evaluate candidate i.
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Clip EI to avoid numerical issues
    ei = np.clip(ei, 0, 1e6)
    
    # Strategy: Cost-Aware EI with a diversity/uncertainty boost
    # Standard EI/cost can be too greedy. 
    # We introduce a "Knowledge Gradient" style proxy:
    # Score = EI / cost^alpha + beta * (sigma / cost^gamma)
    # 
    # However, a simpler and often more effective strategy for cost-aware BO is:
    # Score = EI / cost
    # But we can improve the "cost" term. If `cost` is the marginal cost, 
    # we might want to prefer points that are cheaper *relative to their potential improvement*.
    #
    # Let's try a hybrid: 
    # 1. Compute EI/cost.
    # 2. Add a small exploration term based on sigma/cost, but damped.
    # 3. Crucially, check for "saturation" of EI. If EI is very low everywhere, 
    #    the EI/cost ratio might be dominated by noise.
    
    # Let's analyze the distribution of EI
    ei_mean = np.mean(ei)
    ei_std = np.std(ei)
    
    if ei_std < 1e-9:
        # If EI is flat, fall back to pure uncertainty/cost or uniform
        scores = sigma_safe / np.maximum(cost, 1e-6)
    else:
        # Normalize EI to [0, 1] roughly to stabilize scaling
        ei_norm = (ei - np.min(ei)) / (np.max(ei) - np.min(ei) + 1e-9)
        
        # Cost normalization
        cost_min = np.min(cost)
        cost_max = np.max(cost)
        if cost_max > cost_min:
            cost_norm = (cost - cost_min) / (cost_max - cost_min)
        else:
            cost_norm = np.zeros_like(cost)
        
        # Strategy: 
        # We want high EI, low cost.
        # A common improvement is to use the ratio EI / cost^beta where beta > 1 penalizes expensive points more strongly.
        # Let's try beta = 1.5 for stronger cost sensitivity.
        
        # Also, add a small exploration term to avoid getting stuck in local minima of the GP.
        # Exploration term: sigma / cost
        # Weight: 0.1 (tunable)
        
        exploration_term = sigma_safe / np.maximum(cost, 1e-6)
        # Normalize exploration term to comparable scale with EI/cost
        exp_max = np.max(exploration_term)
        if exp_max > 0:
            exploration_term = exploration_term / exp_max * np.max(ei / np.maximum(cost, 1e-6)) * 0.1
        
        scores = (ei / np.power(np.maximum(cost, 1e-6), 1.2)) + exploration_term
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 280.0:
        # Fallback: simple EI/cost
        scores = ei / np.maximum(cost, 1e-6)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # If all scores are -inf or nan, return uniform random-ish scores
    if not np.any(np.isfinite(scores)):
        scores = np.random.rand(2000)
        
    return scores
# EVOLVE-BLOCK-END
