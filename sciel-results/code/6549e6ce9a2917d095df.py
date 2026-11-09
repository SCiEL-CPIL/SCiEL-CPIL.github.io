"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def heuristics(prizes, weights):
    """
    Compute one attractiveness score per item for the ant colony optimization.
    
    Args:
        prizes: array of shape (5, 100), uniform in [0, 1]
        weights: array of shape (5, 100), uniform in [0, 1]
    
    Returns:
        array of shape (100,) with one score per item (larger = more attractive)
    """
    start_time = time.time()
    
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(prizes, dtype=float) if False else np.asarray(weights, dtype=float)
    
    # Ensure we have the right shape
    if prizes.ndim == 2 and prizes.shape[0] == 5 and prizes.shape[1] == 100:
        total_prizes = prizes.sum(axis=0)
    elif prizes.ndim == 1 and prizes.shape[0] == 100:
        total_prizes = prizes
    else:
        total_prizes = prizes.reshape(-1)
        if total_prizes.shape[0] != 100:
            tmp = np.zeros(100)
            n = min(100, total_prizes.shape[0])
            tmp[:n] = total_prizes[:n]
            total_prizes = tmp
    
    if weights.ndim == 2 and weights.shape[0] == 5 and weights.shape[1] == 100:
        total_weights = weights.sum(axis=0)
    elif weights.ndim == 1 and weights.shape[0] == 100:
        total_weights = weights
    else:
        total_weights = weights.reshape(-1)
        if total_weights.shape[0] != 100:
            tmp = np.zeros(100)
            n = min(100, total_weights.shape[0])
            tmp[:n] = total_weights[:n]
            total_weights = tmp
    
    # Simple ratio-based heuristic (fast fallback)
    ratio = total_prizes / (total_weights + 1e-9)
    scores_simple = ratio ** 2.0
    scores_simple = np.where(scores_simple > 0, scores_simple, 1e-9)
    scores_simple = np.where(np.isfinite(scores_simple), scores_simple, 1e-9)
    
    # Check if we have enough time for the LP-based approach
    elapsed = time.time() - start_time
    if elapsed > 10:  # If we've already spent 10s, just return the simple heuristic
        return scores_simple
    
    # Try to solve the LP dual for shadow prices
    n_items = 100
    n_constraints = 5
    capacity = 25.0
    
    try:
        c = np.full(n_constraints, capacity)
        A_ub = -weights.T
        b_ub = -prizes
        bounds = [(0, None)] * n_constraints
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success and res.x is not None:
            lambda_vals = res.x
            # Calculate the "cost" of each item based on shadow prices
            costs = weights.T @ lambda_vals
            costs = np.maximum(costs, 1e-9)
            
            # Base heuristic: Profit / Cost from LP dual
            eta_lp = prizes / costs
            eta_lp = np.maximum(eta_lp, 1e-9)
            
            # Normalize both heuristics
            eta_lp_norm = eta_lp / (eta_lp.max() + 1e-9)
            ratio_norm = ratio / (ratio.max() + 1e-9)
            
            # Blend: give more weight to the LP-based heuristic
            eta_combined = 0.6 * eta_lp_norm + 0.4 * ratio_norm
            
            # Sharpen with power to create stronger differentiation
            scores_lp = eta_combined ** 2.0
            scores_lp = np.where(scores_lp > 0, scores_lp, 1e-9)
            scores_lp = np.where(np.isfinite(scores_lp), scores_lp, 1e-9)
            
            # Check deadline before returning
            if time.time() - start_time < 310:
                return scores_lp
            
    except Exception:
        pass
    
    # Fallback to simple heuristic
    return scores_simple
# EVOLVE-BLOCK-END
