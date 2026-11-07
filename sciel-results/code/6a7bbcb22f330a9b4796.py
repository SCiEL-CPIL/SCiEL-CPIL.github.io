import numpy as np
import time
from scipy.stats import norm

_start_time = None
_deadline = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Hartmann 6D minimization.
    Uses Pareto frontier selection between Expected Improvement and Cost Efficiency.
    """
    global _start_time, _deadline
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 320.0  # 10s safety margin before 330s kill
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 310:
        # Emergency fallback: simple EI
        eps = 1e-10
        sigma_safe = np.maximum(sigma, eps)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return ei
    
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # 1. Compute Expected Improvement (EI) for minimization
    # EI = E[max(f_best - f, 0)]
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. Compute Cost Efficiency
    # Lower cost is better, so 1/cost is the efficiency measure
    cost_eff = 1.0 / (cost + eps)
    
    # 3. Normalize both objectives to [0, 1] for comparable scales
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range < eps:
        ei_norm = np.ones_like(ei)
    else:
        ei_norm = (ei - ei_min) / ei_range
        
    ce_min = np.min(cost_eff)
    ce_max = np.max(cost_eff)
    ce_range = ce_max - ce_min
    if ce_range < eps:
        ce_norm = np.ones_like(cost_eff)
    else:
        ce_norm = (cost_eff - ce_min) / ce_range
    
    # 4. Pareto Frontier Selection
    # A point is Pareto optimal if no other point is better in both EI and Cost Efficiency.
    # We want to select from the Pareto front, preferring points with higher combined value.
    
    n = len(ei_norm)
    
    # Sort by EI_norm descending (primary objective)
    sort_idx = np.argsort(ei_norm)[::-1]
    sorted_ei = ei_norm[sort_idx]
    sorted_ce = ce_norm[sort_idx]
    
    # Find Pareto front: in sorted order by EI descending,
    # a point is Pareto if its CE is greater than all previous points' CE
    is_pareto = np.zeros(n, dtype=bool)
    max_ce_so_far = -np.inf
    
    for i in range(n):
        if sorted_ce[i] > max_ce_so_far + 1e-12:
            is_pareto[sort_idx[i]] = True
            max_ce_so_far = sorted_ce[i]
    
    # 5. Assign scores: Pareto points get high scores ranked by combined value,
    #    non-Pareto points get low scores
    final_scores = np.zeros(n)
    
    if np.any(is_pareto):
        pareto_mask = is_pareto
        pareto_indices = np.where(pareto_mask)[0]
        
        # Among Pareto points, rank by combined score (EI + CE efficiency)
        # This ensures we pick the best trade-off point
        combined_pareto = ei_norm[pareto_indices] + ce_norm[pareto_indices]
        
        # Scale Pareto scores to be clearly above non-Pareto
        # Use a ranking-based approach: top Pareto points get highest scores
        pareto_rank = np.argsort(combined_pareto)[::-1]  # best first
        pareto_scores = np.zeros(len(pareto_indices))
        
        # Assign scores based on rank with decay
        # Top point gets 10.0, decaying to 5.0 for the worst Pareto point
        if len(pareto_indices) > 1:
            rank_frac = np.arange(len(pareto_indices)) / (len(pareto_indices) - 1)
            pareto_scores = 5.0 + 5.0 * (1.0 - rank_frac)
        else:
            pareto_scores = np.array([10.0])
        
        # Assign in the correct order
        for i, idx in enumerate(pareto_indices):
            final_scores[idx] = pareto_scores[i]
        
        # Non-Pareto points get a low score based on how close they are to the front
        # Use their EI_norm as a soft penalty (they were dominated)
        non_pareto_mask = ~pareto_mask
        if np.any(non_pareto_mask):
            final_scores[non_pareto_mask] = 0.5 * ei_norm[non_pareto_mask]
    else:
        # Fallback: no Pareto points found (shouldn't happen)
        final_scores = ei_norm + ce_norm
    
    # Ensure finite values
    final_scores = np.where(np.isfinite(final_scores), final_scores, 0.0)
    
    return final_scores
