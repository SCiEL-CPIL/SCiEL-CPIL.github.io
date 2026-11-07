import numpy as np
import time
import cma

def _simulate_aco(prizes, distance_matrix, max_len, heuristic, num_ants=20, num_iters=30, seed=42):
    """
    Simulates the ACO process to evaluate a heuristic.
    Deterministic because seed is fixed.
    """
    n = len(prizes)
    H = np.where(heuristic > 0, heuristic, 1e-9)
    
    best_prize = 0
    rng = np.random.default_rng(seed)
    
    d_to_start = distance_matrix[:, 0]
    
    for it in range(num_iters):
        best_iter_prize = 0
        
        for _ in range(num_ants):
            current_node = 0
            visited_mask = np.zeros(n, dtype=bool)
            visited_mask[0] = True
            path_len = 0.0
            total_prize = prizes[0]
            
            while True:
                remaining_budget = max_len - path_len
                d_from_current = distance_matrix[current_node, :]
                valid_indices = np.where(
                    (~visited_mask) & 
                    (d_from_current + d_to_start <= remaining_budget)
                )[0]
                
                if len(valid_indices) == 0:
                    break
                
                h_vals = H[current_node, valid_indices]
                
                total_h = np.sum(h_vals)
                if total_h <= 0:
                    break
                    
                probs = h_vals / total_h
                
                next_node = valid_indices[rng.choice(len(valid_indices), p=probs)]
                
                path_len += distance_matrix[current_node, next_node]
                total_prize += prizes[next_node]
                visited_mask[next_node] = True
                current_node = next_node
            
        best_iter_prize = max(best_iter_prize, total_prize)
        best_prize = max(best_prize, best_iter_prize)

    return best_prize

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 280  # 50s safety margin for 330s limit
    
    n = len(prizes)
    eps = 1e-9
    
    # 1. Generate a good initial heuristic using a parametric model (the old strategy's best guess)
    # This gives CMA-ES a strong starting point, better than random.
    # H[i, j] = (prizes[j] + 0.1) / (distance_matrix[i, j] + 0.1)
    p_col = np.tile(prizes + 0.1, (n, 1))
    d_mat = distance_matrix + 0.1
    init_h = p_col / d_mat
    np.fill_diagonal(init_h, 1e-9)
    
    # Convert to log space for optimization to ensure positivity and multiplicative updates
    # Clip to avoid log(0)
    init_log_h = np.log(np.maximum(init_h, 1e-9))
    init_log_h[~np.isfinite(init_log_h)] = -20.0 # log(1e-9) approx
    
    # Flatten for CMA-ES
    x0 = init_log_h.flatten()
    
    # Objective function for CMA-ES
    def objective(x):
        if time.time() > deadline:
            return 1e10 # Penalize timeout
        log_h = x.reshape((n, n))
        h = np.exp(log_h)
        # Ensure valid range
        h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        np.fill_diagonal(h, 1e-9)
        
        score = _simulate_aco(prizes, distance_matrix, max_len, h)
        return -score # Minimize negative score
    
    # Run CMA-ES
    # n=50, so dimension is 2500. This is high for CMA-ES, but with a good init and limited iterations, it can work.
    # We must limit population size and iterations strictly.
    # Standard CMA-ES pop size ~ 4 + 3*ln(n) ~ 4 + 3*7.8 ~ 27.
    # Let's use a smaller pop size to fit more iterations in the time budget.
    popsize = 15
    
    # Set options
    opts = {
        'maxiter': 50, # Max 50 generations
        'verbose': -9, # Suppress output
        'seed': 42,
        'bounds': [-30, 30] # log(1e-13) to log(1e13), wide range
    }
    
    best_score = -objective(x0)
    best_x = x0.copy()
    
    try:
        es = cma.CMAEvolutionStrategy(x0, 0.1, opts) # sigma=0.1 initial step
        while not es.stop() and time.time() < deadline - 10:
            solutions = es.ask()
            fvals = []
            for sol in solutions:
                f = objective(sol)
                fvals.append(f)
                if f < -best_score:
                    best_score = -f
                    best_x = sol
            es.tell(solutions, fvals)
            # Check time inside loop as well
            if time.time() > deadline - 5:
                break
    except Exception:
        pass

    # Reconstruct best heuristic
    best_log_h = best_x.reshape((n, n))
    best_h = np.exp(best_log_h)
    best_h = np.where(np.isfinite(best_h) & (best_h > 0), best_h, 1e-9)
    np.fill_diagonal(best_h, 1e-9)
    
    return best_h
