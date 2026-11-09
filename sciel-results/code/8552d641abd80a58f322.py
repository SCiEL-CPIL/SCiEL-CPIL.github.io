import numpy as np
import time
from scipy.optimize import minimize

def _run_aco_fast(prizes, dist, max_len, heuristic_matrix, seed, n_iterations=30, n_ants=20):
    """
    Optimized ACO simulation.
    Returns the highest total prize collected by any ant across all iterations.
    """
    n = len(prizes)
    np.random.seed(seed)
    pheromone = np.ones((n, n))
    best_global = 0.0
    
    # Pre-compute valid transitions for each node? 
    # No, because 'visited' changes. But we can pre-compute feasibility based on distance only?
    # Feasibility: length_so_far + d[i,j] + d[j,0] <= max_len.
    # Since length_so_far increases, if a node is infeasible at step k, it remains infeasible for later steps from the same node?
    # Not necessarily, because we move to a different node.
    # However, d[i,j] + d[j,0] is constant for a pair (i,j).
    # We can pre-compute a mask for each i: feasible_j[i] = d[i,j] + d[j,0] <= max_len.
    # Then in the loop, we just check if length_so_far + d[i,j] <= max_len - d[j,0].
    # Actually, the condition is length + d[i,j] + d[j,0] <= max_len.
    # So d[i,j] <= max_len - d[j,0] - length.
    
    feasible_from_i = []
    for i in range(n):
        # For each i, find j such that d[i,j] + d[j,0] <= max_len
        # This is a necessary condition. The sufficient condition depends on current length.
        mask = (dist[i, :] + dist[0, :] <= max_len)
        feasible_from_i.append(np.where(mask)[0])
        
    for _ in range(n_iterations):
        best_iter = 0.0
        
        for _ant in range(n_ants):
            current = 0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            length = 0.0
            prize_sum = prizes[0]
            
            while True:
                # Get candidates
                # candidates must be unvisited and satisfy length + dist[current, j] + dist[j, 0] <= max_len
                # We can filter feasible_from_i[current] by visited and the dynamic length constraint.
                
                potential_js = feasible_from_i[current]
                if len(potential_js) == 0:
                    break
                
                # Filter by visited
                unvisited_mask = ~visited[potential_js]
                if not np.any(unvisited_mask):
                    break
                    
                valid_js = potential_js[unvisited_mask]
                
                # Filter by length constraint: dist[current, j] <= max_len - dist[j, 0] - length
                # dist[current, valid_js] + dist[0, valid_js] <= max_len - length
                d_cj = dist[current, valid_js]
                d_j0 = dist[0, valid_js]
                length_mask = (d_cj + d_j0) <= (max_len - length)
                
                if not np.any(length_mask):
                    break
                    
                final_js = valid_js[length_mask]
                
                # Calculate probabilities
                # p = pheromone[current, j] * heuristic[current, j]
                # Handle non-finite or non-positive heuristic
                h_vals = heuristic_matrix[current, final_js]
                # Sanitize h_vals
                h_vals = np.where(np.isfinite(h_vals) & (h_vals > 0), h_vals, 1e-9)
                
                p_vals = pheromone[current, final_js] * h_vals
                
                total_p = np.sum(p_vals)
                if total_p == 0:
                    break
                
                probs = p_vals / total_p
                
                # Sample
                r = np.random.rand()
                cumsum = np.cumsum(probs)
                chosen_idx = np.searchsorted(cumsum, r)
                if chosen_idx >= len(final_js):
                    chosen_idx = len(final_js) - 1
                
                next_node = final_js[chosen_idx]
                
                length += dist[current, next_node]
                visited[next_node] = True
                prize_sum += prizes[next_node]
                current = next_node
        
        best_iter = max(best_iter, prize_sum)
        
        pheromone = pheromone * 0.9
        pheromone += best_iter / 10.0
        
        if best_iter > best_global:
            best_global = best_iter
            
    return best_global

def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # Define a set of base heuristic generators
    # We will optimize weights w such that H = sum(w_k * H_k)
    
    d = distance_matrix
    p = prizes
    
    # Base heuristics
    # 1. Prize / (dist + eps)
    h1 = (p[None, :] + 1e-9) / (d + 1e-9)
    # 2. Prize / (dist + dist_to_0 + eps)
    d_0 = d[0, :]
    h2 = (p[None, :] + 1e-9) / (d + d_0[None, :] + 1e-9)
    # 3. Prize * exp(-dist)
    h3 = (p[None, :] + 1e-9) * np.exp(-d)
    # 4. Prize / (dist^2 + eps)
    h4 = (p[None, :] + 1e-9) / (d + 1e-9)**2
    # 5. Prize / (dist + dist_i0 + dist_j0)
    h5 = (p[None, :] + 1e-9) / (d + d_0[:, None] + d_0[None, :] + 1e-9)
    
    # Stack them: shape (5, n, n)
    base_heuristics = np.stack([h1, h2, h3, h4, h5], axis=0)
    
    # Ensure base heuristics are sanitized for the linear combination step
    # Actually, we sanitize the final combined matrix.
    
    test_seeds = [42, 123, 999, 7, 12]
    
    def objective(weights):
        """
        Minimize negative mean ACO score.
        weights: array of shape (5,)
        """
        # Combine heuristics
        # H = sum(w_k * H_k)
        # Weights should be non-negative? The problem says "larger is more attractive".
        # Negative weights would make some edges unattractive, which might be good for pruning, 
        # but the ACO code clamps non-positive to 1e-9. 
        # So negative weights effectively zero out those contributions.
        # Let's allow any real weights, but the clamping in ACO handles the rest.
        
        h_mat = np.tensordot(weights, base_heuristics, axes=(0, 0))
        
        # Sanitize
        h_mat = np.where(np.isfinite(h_mat) & (h_mat > 0), h_mat, 1e-9)
        
        scores = []
        for seed in test_seeds:
            try:
                score = _run_aco_fast(prizes, distance_matrix, max_len, h_mat, seed)
                scores.append(score)
            except Exception:
                scores.append(0.0)
        
        mean_score = np.mean(scores)
        return -mean_score
    
    # Initial guess: equal weights
    w0 = np.ones(5) / 5.0
    
    # Nelder-Mead options
    options = {
        'maxiter': 50,
        'maxfev': 100,
        'xatol': 1e-3,
        'fatol': 1e-3
    }
    
    best_w = w0.copy()
    best_val = objective(w0)
    
    # Check time
    if time.time() > deadline - 60:
        # Not enough time for optimization, return a simple heuristic
        h_mat = base_heuristics[1] # h2 is usually good
        return np.where(np.isfinite(h_mat) & (h_mat > 0), h_mat, 1e-9)
        
    try:
        res = minimize(objective, w0, method='Nelder-Mead', options=options)
        
        if res.fun < best_val:
            best_w = res.x
            best_val = res.fun
    except Exception:
        pass
        
    # Construct final heuristic
    h_mat = np.tensordot(best_w, base_heuristics, axes=(0, 0))
    h_mat = np.where(np.isfinite(h_mat) & (h_mat > 0), h_mat, 1e-9)
    
    return h_mat
