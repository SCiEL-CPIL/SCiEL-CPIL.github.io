import numpy as np
import time

def _aco_soft_min_score(distance_matrix, heuristic, n_ants=10, n_iterations=20, rho=0.9, seed=42):
    """
    Runs an ACO simulation with a fixed seed and returns a smooth surrogate score.
    The score is the soft-min (log-sum-exp based) of the tour lengths from the last iteration.
    This is smoother than min(tour_lengths) and provides a better gradient signal for beta optimization.
    """
    n = distance_matrix.shape[0]
    tau = np.ones((n, n))
    
    # Use a specific RNG for the simulation to ensure consistency
    rng = np.random.default_rng(seed)
    
    best_tour_len = float('inf')
    last_tour_lengths = []
    
    for it in range(n_iterations):
        tour_lengths = []
        for ant in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            tour_len = 0.0
            tour_path = [0]
            
            for _ in range(n - 1):
                probs = tau[current, :] * heuristic[current, :]
                probs[visited] = 0.0
                
                total_prob = probs.sum()
                if total_prob < 1e-10:
                    unvisited_indices = np.where(~visited)[0]
                    next_node = unvisited_indices[rng.integers(len(unvisited_indices))]
                else:
                    probs = probs / total_prob
                    next_node = rng.choice(n, p=probs)
                
                tour_len += distance_matrix[current, next_node]
                visited[next_node] = True
                tour_path.append(next_node)
                current = next_node
            
            tour_len += distance_matrix[current, 0]
            tour_lengths.append(tour_len)
            
            deposit = 1.0 / tour_len
            for i in range(n):
                u = tour_path[i]
                v = tour_path[(i + 1) % n]
                tau[u, v] += deposit
            
        tau *= rho
        
        if tour_lengths:
            current_best = min(tour_lengths)
            if current_best < best_tour_len:
                best_tour_len = current_best
            last_tour_lengths = tour_lengths

    # Calculate soft-min score from the last iteration's tour lengths
    # soft_min(x) = -log(sum(exp(-x))) is not quite right for minimization directly in log space
    # We want a value that is close to min(x) but smooth.
    # log-sum-exp(x) = max(x) + log(sum(exp(x - max(x))))
    # For minimization, we can use -log-sum-exp(-x)
    if last_tour_lengths:
        x = np.array(last_tour_lengths)
        max_x = np.max(x)
        # Stable log-sum-exp for minimization
        # LSE(-x) = max(-x) + log(sum(exp(-x - max(-x)))) = -min(x) + log(sum(exp(min(x) - x)))
        # soft_min = -LSE(-x) = min(x) - log(sum(exp(min(x) - x)))
        min_x = np.min(x)
        exp_terms = np.exp(min_x - x)
        sum_exp = np.sum(exp_terms)
        soft_min_val = min_x - np.log(sum_exp)
        return soft_min_val
    
    return best_tour_len

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 280.0
    
    n = distance_matrix.shape[0]
    
    # Base heuristic: inverse distance
    safe_dist = distance_matrix.copy()
    np.fill_diagonal(safe_dist, 1.0)
    safe_dist[safe_dist < 1e-9] = 1e-9
    
    base_heur = 1.0 / safe_dist
    
    # We will optimize the exponent beta such that heuristic = base_heur ** beta
    # This is equivalent to heuristic = 1 / (dist ** beta)
    
    best_beta = 2.0
    best_score = float('inf')
    best_heuristic = base_heur.copy()
    
    # Coarse search over a wide range of beta values
    # Beta > 0 favors shorter edges. Beta < 0 would favor longer edges (unlikely to be good).
    # Typical TSP heuristics use beta around 1-5.
    betas = np.linspace(0.5, 6.0, 25)
    
    for beta in betas:
        if time.time() > deadline - 30.0:
            break
        
        h = base_heur ** beta
        h[h <= 0] = 1e-9
        
        # Use soft-min score for smoother optimization signal
        score = _aco_soft_min_score(distance_matrix, h, n_ants=10, n_iterations=20)
        
        if score < best_score:
            best_score = score
            best_beta = beta
            best_heuristic = h.copy()
    
    # Fine-grained search around the best beta found
    if time.time() < deadline - 40.0:
        lower = max(0.1, best_beta - 0.5)
        upper = min(10.0, best_beta + 0.5)
        fine_betas = np.linspace(lower, upper, 20)
        
        for beta in fine_betas:
            if time.time() > deadline - 20.0:
                break
            
            h = base_heur ** beta
            h[h <= 0] = 1e-9
            score = _aco_soft_min_score(distance_matrix, h, n_ants=10, n_iterations=20)
            
            if score < best_score:
                best_score = score
                best_beta = beta
                best_heuristic = h.copy()
                
    # Final polish: try a few more precise points if time allows
    if time.time() < deadline - 10.0:
        # Interpolate slightly around the best beta
        delta_beta = 0.1
        candidates = [best_beta - delta_beta, best_beta, best_beta + delta_beta]
        
        for beta in candidates:
            if time.time() > deadline - 5.0:
                break
            if beta <= 0: continue
            
            h = base_heur ** beta
            h[h <= 0] = 1e-9
            score = _aco_soft_min_score(distance_matrix, h, n_ants=10, n_iterations=20)
            
            if score < best_score:
                best_score = score
                best_beta = beta
                best_heuristic = h.copy()

    # Final sanity checks
    best_heuristic[best_heuristic <= 0] = 1e-9
    best_heuristic[~np.isfinite(best_heuristic)] = 1e-9
    
    return best_heuristic
