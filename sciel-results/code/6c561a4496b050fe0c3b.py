import numpy as np
from scipy.optimize import linprog

def heuristics(prizes, weights):
    """
    Computes heuristic values for items in a multiple-constraint knapsack problem.
    Uses shadow prices from the LP relaxation dual to estimate item attractiveness.
    """
    n_items = weights.shape[1]
    n_constraints = weights.shape[0]
    capacity = 25.0
    
    # We solve the dual of the continuous relaxation (ignoring x_i <= 1 for simplicity 
    # in the dual variable extraction, or using the full dual if needed).
    # The simple dual: Min sum(lambda_j * C_j) s.t. sum(lambda_j * w_ji) >= p_i, lambda_j >= 0
    # This gives lower bounds on the shadow prices.
    
    c = np.full(n_constraints, capacity)
    A_ub = -weights.T 
    b_ub = -prizes
    bounds = [(0, None)] * n_constraints
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            lambda_vals = res.x
        else:
            lambda_vals = np.ones(n_constraints)
    except Exception:
        lambda_vals = np.ones(n_constraints)
        
    # Calculate the "cost" of each item based on shadow prices
    costs = weights.T @ lambda_vals
    
    # Prevent division by zero
    costs = np.maximum(costs, 1e-9)
    
    # Heuristic: Profit / Cost
    eta = prizes / costs
    
    # Ensure positivity
    eta = np.maximum(eta, 1e-9)
    
    return eta
