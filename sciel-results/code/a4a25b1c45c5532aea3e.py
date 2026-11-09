import numpy as np
import time

def _solve_lagrangian_dual(prizes, weights, capacity=25, max_iter=200, step_size=0.1):
    """
    Solve the Lagrangian dual problem to find optimal dual prices.
    """
    num_constraints, n_items = prizes.shape
    total_prize = prizes.sum(axis=0)
    
    lambda_vec = np.zeros(num_constraints)
    
    for _ in range(max_iter):
        # Reduced cost for each item
        reduced_costs = total_prize - weights.T @ lambda_vec
        
        # Select items with positive reduced cost
        selected = reduced_costs > 0
        
        # Compute usage
        usage = weights[:, selected].sum(axis=1)
        
        # Subgradient of the dual function
        # We are minimizing g(lambda) = sum_j max(0, prize_j - lambda @ w_j) + lambda @ C
        # Subgradient = C - usage (for lambda > 0) or C - usage (if lambda=0, one-sided)
        # To minimize, we update lambda -= step * subgrad
        # lambda_new = max(0, lambda_old - step * (C - usage)) = max(0, lambda_old + step * (usage - C))
        
        subgrad = capacity - usage
        lambda_vec = np.maximum(0, lambda_vec - step_size * subgrad)
        
        # Check convergence
        if np.max(np.abs(subgrad)) < 0.01:
            break
            
    return lambda_vec

def _compute_heuristics(prizes, weights, deadline=None):
    start_time = time.time()
    if deadline is None:
        deadline = start_time + 290
        
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    num_constraints, n_items = prizes.shape
    
    # Solve Lagrangian dual to get optimal dual prices
    dual_prices = _solve_lagrangian_dual(prizes, weights, capacity=25, max_iter=200, step_size=0.1)
    
    # Compute reduced costs: prize - dual_price @ weight
    total_prize = prizes.sum(axis=0)
    reduced_costs = total_prize - weights.T @ dual_prices
    
    # The heuristic should guide ants towards items with high reduced cost.
    # However, the ACO algorithm multiplies pheromone * heuristic.
    # If we use reduced_costs directly, items with negative reduced cost get low score.
    # We need to map reduced costs to positive values.
    
    # Strategy: Use a soft mapping.
    # Items with high positive reduced cost are very attractive.
    # Items with negative reduced cost are less attractive.
    
    # Normalize reduced costs to [0, 1] range?
    # Or use exp(reduced_cost)?
    
    # Let's try: heuristic = reduced_costs - min(reduced_costs) + 1e-9
    # This preserves the relative ordering and makes them all positive.
    
    min_rc = reduced_costs.min()
    heuristic = reduced_costs - min_rc + 1e-9
    
    # Ensure finite
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic

def heuristics(prizes, weights):
    start_time = time.time()
    try:
        result = _compute_heuristics(prizes, weights, start_time + 290)
        result = np.asarray(result, dtype=np.float64)
        
        if result.ndim != 1:
            raise ValueError("Heuristic must be a 1D array")
        
        if not np.all(np.isfinite(result)):
            raise ValueError("Heuristic must contain only finite values")
        
        prizes_arr = np.asarray(prizes)
        expected_n_items = prizes_arr.shape[1] if prizes_arr.ndim == 2 else len(prizes_arr)
        
        if result.shape[0] != expected_n_items:
            raise ValueError(f"Heuristic shape {result.shape} doesn't match expected ({expected_n_items},)")
        
        return result
        
    except Exception:
        prizes_arr = np.asarray(prizes, dtype=np.float64)
        if prizes_arr.ndim == 2:
            n_items = prizes_arr.shape[1]
        else:
            n_items = len(prizes_arr)
        return np.ones(n_items, dtype=np.float64)
