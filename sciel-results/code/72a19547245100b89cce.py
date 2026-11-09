"""Heuristic design for ACO-based multiple-constraint knapsack."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def _simulate_aco(prizes, weights, heuristic, ants=20, iters=30, capacity=25, seed=42):
    """
    Run a small ACO simulation to evaluate the quality of a heuristic.
    Returns the best total prize found.
    """
    n_items = len(prizes)
    n_constraints = weights.shape[0]
    
    eta = np.where(heuristic > 0, heuristic, 1e-9)
    
    tau = np.ones(n_items)
    best_prize_overall = 0.0
    
    rng = np.random.RandomState(seed)
    
    for _ in range(iters):
        best_prize_this_iter = 0.0
        
        for _ in range(ants):
            current_weights = np.zeros(n_constraints)
            total_prize = 0.0
            used_items = np.zeros(n_items, dtype=bool)
            
            while True:
                remaining_cap = capacity - current_weights
                fits_mask = np.all(weights.T <= remaining_cap, axis=1)
                fits_mask[used_items] = False
                
                if not np.any(fits_mask):
                    break
                
                candidate_indices = np.where(fits_mask)[0]
                candidate_tau = tau[candidate_indices]
                candidate_eta = eta[candidate_indices]
                
                probs = candidate_tau * candidate_eta
                sum_probs = np.sum(probs)
                
                if sum_probs == 0:
                    break
                
                probs = probs / sum_probs
                
                selected_idx = rng.choice(len(candidate_indices), p=probs)
                item_idx = candidate_indices[selected_idx]
                
                total_prize += prizes[item_idx]
                current_weights += weights[:, item_idx]
                used_items[item_idx] = True
            
        if total_prize > best_prize_this_iter:
            best_prize_this_iter = total_prize
            
        tau = tau * 0.9 + best_prize_this_iter / 10.0
        
        if best_prize_this_iter > best_prize_overall:
            best_prize_overall = best_prize_this_iter
            
    return best_prize_overall

def _lp_relaxation_heuristic(prizes, weights, capacity=25):
    """
    Solve LP relaxation of MCKP to get dual prices (shadow prices).
    The heuristic is prize - sum(dual * weight), which is the reduced cost.
    Items with high reduced cost are attractive.
    """
    n_constraints = weights.shape[0]
    n_items = len(prizes)
    
    # LP: maximize prizes @ x subject to weights @ x <= capacity, 0 <= x <= 1
    # linprog minimizes, so we negate the objective
    c = -prizes
    A_ub = weights  # shape (n_constraints, n_items)
    b_ub = np.full(n_constraints, capacity)
    bounds = [(0, 1)] * n_items
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            # Get dual values (shadow prices) from the constraints
            # In HiGHS, duals are in res.ineqlin.marginals or similar
            # For A_ub constraints, the duals are -res.ineqlin.marginals (since we maximize)
            # Actually, for minimization with A_ub x <= b_ub, the dual y >= 0
            # The reduced cost for variable j is c_j - A_ub[:, j] @ y
            # Since we're maximizing prizes, the "value" of item j is prizes[j] - weights[:, j] @ y
            # where y are the shadow prices (duals)
            
            # In scipy linprog with HiGHS, for A_ub constraints:
            # res.ineqlin.marginals gives the dual values for the inequality constraints
            # These are the shadow prices (lambda)
            duals = np.abs(res.ineqlin.marginals)  # shadow prices, should be >= 0
            
            # Heuristic: reduced cost = prize - sum(dual * weight)
            # This is how much "extra" value an item provides after accounting for resource costs
            reduced_costs = prizes - weights @ duals
            
            # Ensure positive
            heuristic = np.where(reduced_costs > 0, reduced_costs, 1e-9)
            
            # Also add a small component based on prize/weight ratio for items with low reduced cost
            # to ensure diversity
            ratio = prizes / (weights.sum(axis=0) + 1e-9)
            heuristic = heuristic + 0.1 * ratio
            
            return np.where(heuristic > 0, heuristic, 1e-9)
    except Exception:
        pass
    
    # Fallback: prize / sum(weights)
    return prizes / (weights.sum(axis=0) + 1e-9)

def _greedy_local_search(prizes, weights, capacity=25, max_iters=5000):
    """
    Greedy construction + local search to find a good feasible solution.
    Returns the heuristic derived from the solution.
    """
    n_items = len(prizes)
    n_constraints = weights.shape[0]
    
    # Greedy by prize/weight ratio
    ratios = prizes / (weights.sum(axis=0) + 1e-9)
    order = np.argsort(-ratios)
    
    selected = np.zeros(n_items, dtype=bool)
    current_weights = np.zeros(n_constraints)
    total_prize = 0.0
    
    for idx in order:
        if np.all(weights[:, idx] <= capacity - current_weights):
            selected[idx] = True
            current_weights += weights[:, idx]
            total_prize += prizes[idx]
    
    # Local search: try to swap out low-value items for high-value ones
    improved = True
    while improved and max_iters > 0:
        improved = False
        max_iters -= 1
        
        selected_indices = np.where(selected)[0]
        unselected_indices = np.where(~selected)[0]
        
        for out_idx in selected_indices:
            for in_idx in unselected_indices:
                new_weights = current_weights - weights[:, out_idx] + weights[:, in_idx]
                if np.all(new_weights <= capacity):
                    new_prize = total_prize - prizes[out_idx] + prizes[in_idx]
                    if new_prize > total_prize:
                        selected[out_idx] = False
                        selected[in_idx] = True
                        current_weights = new_weights
                        total_prize = new_prize
                        improved = True
                        break
            if improved:
                break
    
    # Derive heuristic: items in the solution get high scores, others lower
    heuristic = np.ones(n_items)
    heuristic[selected] = 10.0
    
    # Add ratio component
    heuristic += prizes / (weights.sum(axis=0) + 1e-9)
    
    return np.where(heuristic > 0, heuristic, 1e-9)

def heuristics(prizes, weights):
    start_time = time.time()
    deadline = start_time + 280.0
    
    n_items = len(prizes)
    n_constraints = weights.shape[0]
    
    candidates = []
    
    # 1. LP relaxation based heuristic (dual prices)
    h_lp = _lp_relaxation_heuristic(prizes, weights)
    if np.all(np.isfinite(h_lp)):
        candidates.append(h_lp)
    
    # 2. Greedy local search based
    h_greedy = _greedy_local_search(prizes, weights)
    if np.all(np.isfinite(h_greedy)):
        candidates.append(h_greedy)
    
    # 3. Prize / sum(weights)
    h1 = prizes / (weights.sum(axis=0) + 1e-9)
    candidates.append(h1)
    
    # 4. Prize / max(weights)
    h2 = prizes / (weights.max(axis=0) + 1e-9)
    candidates.append(h2)
    
    # 5. Prize / mean(weights)
    h3 = prizes / (weights.mean(axis=0) + 1e-9)
    candidates.append(h3)
    
    # 6. Prize * (1 - max(weights))
    h4 = prizes * (1 - weights.max(axis=0))
    candidates.append(np.where(h4 > 0, h4, 1e-9))
    
    # 7. h1 ** 2
    h5 = h1 ** 2
    candidates.append(h5)
    
    # 8. Prize / sqrt(sum(weights))
    h6 = prizes / (np.sqrt(weights.sum(axis=0)) + 1e-9)
    candidates.append(h6)
    
    # 9. Geometric mean based
    log_weights = np.log(weights + 1e-9)
    geo_mean = np.exp(log_weights.mean(axis=0))
    h8 = prizes / (geo_mean + 1e-9)
    candidates.append(h8)
    
    # 10. Prize / (max + sum)
    h9 = prizes / (weights.max(axis=0) + weights.sum(axis=0) + 1e-9)
    candidates.append(h9)
    
    # Evaluate candidates with ACO simulation
    best_score = -1
    best_heuristic = candidates[0]
    
    seeds = [42, 123, 456]
    
    for h in candidates:
        if time.time() > deadline:
            break
            
        if not np.all(np.isfinite(h)):
            continue
        h_safe = np.where(h > 0, h, 1e-9)
        
        score = _simulate_aco(prizes, weights, h_safe, seed=seeds[0])
        
        if score > best_score:
            best_score = score
            best_heuristic = h_safe
    
    # If time permits, confirm with a second seed
    if time.time() < deadline - 40:
        confirm_score = _simulate_aco(prizes, weights, best_heuristic, seed=seeds[1])
    
    final_heuristic = np.where(best_heuristic > 0, best_heuristic, 1e-9)
    final_heuristic = np.where(np.isfinite(final_heuristic), final_heuristic, 1e-9)
    
    return final_heuristic
# EVOLVE-BLOCK-END
