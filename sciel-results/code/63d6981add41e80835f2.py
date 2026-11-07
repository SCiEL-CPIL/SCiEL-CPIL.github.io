import numpy as np

def heuristics(prizes, weights):
    """
    Computes heuristic scores for items in a Multiple-Constraint Knapsack Problem.
    Uses a Lagrangian relaxation approach with greedy estimation of constraint prices.
    """
    # Input shapes: (5, 100)
    n_cons, n_items = prizes.shape
    
    # Total prize per item (sum across the 5 dimensions)
    P = np.sum(prizes, axis=0)
    
    # Weights per item, transposed to (100, 5) for easier dot product
    W = weights.T
    
    # Capacity for each constraint
    capacity = 25.0
    
    # Initialize constraint prices (Lagrangian multipliers)
    lambda_c = np.ones(n_cons)
    
    # Iterative greedy estimation of lambda_c
    # We run a few rounds to refine the prices based on which constraints are binding
    for _ in range(20):
        # Calculate effective cost for each item
        # eff_cost[i] = sum_c lambda_c * W[i, c]
        eff_cost = W.dot(lambda_c)
        
        # Avoid division by zero
        eff_cost_safe = eff_cost + 1e-9
        
        # Calculate profit-to-cost ratio
        ratios = P / eff_cost_safe
        
        # Sort items by ratio in descending order
        indices = np.argsort(-ratios)
        
        # Greedily select items that fit
        current_weights = np.zeros(n_cons)
        
        for idx in indices:
            w_item = W[idx]
            # Check if item fits in all constraints
            if np.all(current_weights + w_item <= capacity):
                current_weights += w_item
                
        # Update lambda_c based on binding constraints
        # If a constraint is nearly full, increase its price to discourage future selection
        # If it's loose, we could decrease it, but keeping it or slight decay is safer
        for c in range(n_cons):
            if current_weights[c] > 0.95 * capacity:
                lambda_c[c] *= 1.2
            elif current_weights[c] < 0.5 * capacity:
                # If very loose, reduce price slightly to allow more exploration there
                lambda_c[c] *= 0.9
            else:
                # Keep as is
                pass
                
    # Final heuristic calculation
    eff_cost_final = W.dot(lambda_c) + 1e-9
    heur = P / eff_cost_final
    
    # Sanitize: ensure positive and finite
    heur[~np.isfinite(heur)] = 1e-9
    heur[heur <= 0] = 1e-9
    
    return heur
