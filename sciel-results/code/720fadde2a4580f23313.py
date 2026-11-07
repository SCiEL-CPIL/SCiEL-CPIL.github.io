import numpy as np
import time
from scipy.spatial import cKDTree

def _fast_aco_simulate(distance_matrix, heuristic, n_ants=10, n_iterations=20, rho=0.9, seed=42):
    """
    Fast internal ACO simulation to evaluate heuristic quality.
    Uses a fixed seed for consistent evaluation during local search.
    """
    n = distance_matrix.shape[0]
    tau = np.ones((n, n))
    best_tour_len = float('inf')
    
    # Use a specific RNG for the simulation to ensure consistency
    # The actual evaluator uses its own seed, but for heuristic optimization,
    # we want a stable signal.
    rng = np.random.default_rng(seed)
    
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
                
    return best_tour_len

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    n = distance_matrix.shape[0]
    
    # Base heuristic: inverse distance
    safe_dist = distance_matrix.copy()
    np.fill_diagonal(safe_dist, 1.0)
    safe_dist[safe_dist < 1e-9] = 1e-9
    
    # Initial heuristic: standard 1/d
    current_heur = 1.0 / safe_dist
    
    # Evaluate initial heuristic
    # We use a fixed seed for the internal simulation to get a consistent score
    # for the local search, but the final result is just the matrix.
    current_score = _fast_aco_simulate(distance_matrix, current_heur, n_ants=10, n_iterations=20, seed=123)
    
    best_heur = current_heur.copy()
    best_score = current_score
    
    # Local Search / Hill Climbing on the heuristic matrix
    # We treat the heuristic as a set of continuous variables (log-heuristic)
    # and perform random perturbations to find a better local optimum.
    
    # Work in log-space to maintain positivity
    log_heur = np.log(current_heur)
    
    step_size = 0.1
    no_improve_count = 0
    max_no_improve = 20
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        # Generate a random perturbation
        # Perturb a random subset of entries to keep the search efficient
        num_perturb = max(5, n // 5)
        indices = np.random.choice(n * n, num_perturb, replace=False)
        rows = indices // n
        cols = indices % n
        
        # Skip diagonal
        valid_mask = rows != cols
        if not np.any(valid_mask):
            continue
        
        r_idx = rows[valid_mask]
        c_idx = cols[valid_mask]
        
        # Random step
        steps = np.random.normal(0, step_size, size=len(r_idx))
        
        # Apply perturbation
        new_log_heur = log_heur.copy()
        new_log_heur[r_idx, c_idx] += steps
        
        # Convert back to heuristic
        new_heur = np.exp(new_log_heur)
        
        # Ensure positive
        new_heur[new_heur <= 0] = 1e-9
        
        # Evaluate
        new_score = _fast_aco_simulate(distance_matrix, new_heur, n_ants=10, n_iterations=20, seed=123)
        
        if new_score < best_score:
            best_score = new_score
            best_heur = new_heur.copy()
            log_heur = new_log_heur
            no_improve_count = 0
            # Optionally reduce step size on success (annealing-like)
            step_size *= 0.95
        else:
            no_improve_count += 1
            # Increase step size slightly if stuck? Or keep it?
            # Let's keep it constant or increase slightly to escape local minima
            step_size *= 1.02
            if step_size > 0.5:
                step_size = 0.5

    # Final sanity checks
    best_heur[best_heur <= 0] = 1e-9
    best_heur[~np.isfinite(best_heur)] = 1e-9
    
    return best_heur
