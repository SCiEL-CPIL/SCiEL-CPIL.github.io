"""Acquisition function using Hypervolume-based Pareto front scoring."""
import time
import numpy as np
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Hypervolume contribution of (EI, 1/Cost) Pareto front.
    
    Args:
        mu: array shape (2000,) - GP mean predictions in objective units
        sigma: array shape (2000,) - GP std dev predictions in objective units
        f_best: float - lowest observed value
        cost: array shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Time guard: if we're running out of time, return a simple fallback
    if time.time() > _DEADLINE:
        return np.ones_like(mu) / np.maximum(cost, 1e-9)
    
    eps = 1e-9
    s = np.maximum(sigma, eps)
    
    # Calculate Expected Improvement
    z = (f_best - mu) / s
    z = np.clip(z, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Secondary objective: Efficiency (1/Cost)
    c = np.maximum(cost, eps)
    inv_cost = 1.0 / c
    
    # Identify Pareto front in (EI, InvCost) space
    # We want to maximize both EI and InvCost.
    # Sort by EI descending.
    n = len(ei)
    sorted_idx = np.argsort(-ei)
    
    # Find Pareto optimal points
    pareto_indices = []
    max_inv_cost_so_far = 0.0
    
    for i in sorted_idx:
        if inv_cost[i] > max_inv_cost_so_far:
            pareto_indices.append(i)
            max_inv_cost_so_far = inv_cost[i]
    
    # Calculate Hypervolume contribution for each Pareto point
    # The contribution of a point (x_i, y_i) is the area of the rectangle
    # [x_next, x_i] x [0, y_i] where x_next is the EI of the next Pareto point.
    hv_scores = np.zeros(n)
    
    if len(pareto_indices) > 0:
        # Extract EI and InvCost for Pareto points
        pareto_eis = np.array([ei[i] for i in pareto_indices])
        pareto_ics = np.array([inv_cost[i] for i in pareto_indices])
        
        # Calculate widths
        # The last point dominates down to EI=0
        widths = np.zeros(len(pareto_eis))
        if len(pareto_eis) > 1:
            widths[:-1] = pareto_eis[:-1] - pareto_eis[1:]
        if len(pareto_eis) > 0:
            widths[-1] = pareto_eis[-1]
        
        # Contribution = width * height (InvCost)
        contributions = widths * pareto_ics
        
        # Assign scores to the original indices
        for k, idx in enumerate(pareto_indices):
            hv_scores[idx] = contributions[k]
            
    # To ensure we always have a valid score and prefer points on the front,
    # we return the HV score. Points not on the front get 0.
    # However, if all scores are 0 (e.g. EI is 0 everywhere), we need a fallback.
    if np.max(hv_scores) == 0:
        # Fallback to EI / Cost if no Pareto points found (shouldn't happen if EI>0)
        # Or just return EI
        return ei / c
        
    return hv_scores
