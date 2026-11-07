import numpy as np
import time

def heuristics(prizes, weights):
    """
    Computes heuristic scores for items in a Multiple-Constraint Knapsack Problem.
    Robust to 1D or 2D input shapes.
    """
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Determine n_items and n_cons
    if prizes.ndim == 1:
        n_items = prizes.shape[0]
        n_cons = 1
        P = prizes
        W = weights.reshape(n_items, 1)
        capacity = np.array([25.0])
    elif prizes.ndim == 2:
        d0, d1 = prizes.shape
        if d0 < d1:
            # Shape (cons, items)
            n_cons, n_items = d0, d1
            P = np.sum(prizes, axis=0)
            W = weights.T  # (items, cons)
        else:
            # Shape (items, cons)
            n_items, n_cons = d0, d1
            P = np.sum(prizes, axis=1)
            W = weights    # (items, cons)
        capacity = np.full(n_cons, 25.0)
    else:
        # Fallback for weird shapes
        return np.ones(prizes.size)
        
    if time.time() > deadline:
        return np.ones(n_items)

    # Lagrangian Heuristic
    lambda_c = np.ones(n_cons)
    
    # Iterative refinement
    max_iter = 20
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        eff_cost = W.dot(lambda_c)
        eff_cost_safe = eff_cost + 1e-9
        ratios = P / eff_cost_safe
        
        indices = np.argsort(-ratios)
        current_weights = np.zeros(n_cons)
        
        for idx in indices:
            w_item = W[idx]
            if np.all(current_weights + w_item <= capacity):
                current_weights += w_item
        
        for c in range(n_cons):
            if current_weights[c] > 0.9 * capacity[c]:
                lambda_c[c] *= 1.2
            elif current_weights[c] < 0.5 * capacity[c]:
                lambda_c[c] *= 0.8
            
        # Normalize to prevent numerical issues
        if np.sum(lambda_c) > 0:
            lambda_c = lambda_c / np.sum(lambda_c) * n_cons

    eff_cost_final = W.dot(lambda_c) + 1e-9
    heur = P / eff_cost_final
    
    heur[~np.isfinite(heur)] = 1e-9
    heur[heur <= 0] = 1e-9
    
    return heur
