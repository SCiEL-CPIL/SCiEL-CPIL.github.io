"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _simulate_aco(prizes, distance_matrix, max_len, heuristic, n_ants=20, n_iters=30, seed=42):
    """Run the fixed ACO and return the best total prize collected by any ant."""
    rng = np.random.RandomState(seed)
    n = len(prizes)
    pheromone = np.ones((n, n))
    best_total = 0.0
    
    for it in range(n_iters):
        iter_best = 0.0
        for ant in range(n_ants):
            visited = [0]
            current = 0
            length_so_far = 0.0
            total_prize = 0.0
            
            while True:
                # Find qualifying nodes
                candidates = []
                for j in range(n):
                    if j in visited:
                        continue
                    if length_so_far + distance_matrix[current, j] + distance_matrix[j, 0] <= max_len:
                        candidates.append(j)
                
                if not candidates:
                    break
                
                # Compute probabilities
                probs = np.array([pheromone[current, j] * heuristic[current, j] for j in candidates])
                probs = np.maximum(probs, 1e-9)
                probs /= probs.sum()
                
                # Select next node
                idx = rng.choice(len(candidates), p=probs)
                next_node = candidates[idx]
                
                visited.append(next_node)
                length_so_far += distance_matrix[current, next_node]
                total_prize += prizes[next_node]
                current = next_node
            
            iter_best = max(iter_best, total_prize)
        
        best_total = max(best_total, iter_best)
        
        # Update pheromone
        pheromone *= 0.9
        pheromone += iter_best / 10.0
    
    return best_total


def _make_heuristic(prizes, distance_matrix, max_len, params):
    """Generate heuristic from parameters: (prize_exp, dist_exp, origin_exp, prize_weight, dist_weight, origin_weight)"""
    prize_exp, dist_exp, origin_exp, prize_w, dist_w, origin_w = params
    
    d_to_origin = distance_matrix[0, :]
    
    prize_term = (prizes[None, :] ** prize_exp + 1e-9) * prize_w
    dist_penalty = (1.0 / (distance_matrix + 1e-9) ** dist_exp) * dist_w
    origin_bonus = (1.0 / (d_to_origin[None, :] + 1e-9) ** origin_exp) * origin_w
    
    heuristic = prize_term * dist_penalty * origin_bonus
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic


def _optimize_parameters(prizes, distance_matrix, max_len, initial_params, n_evals=50, time_budget=25.0):
    """Use Nelder-Mead to optimize heuristic parameters."""
    from scipy.optimize import minimize
    
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_budget:
            return 1e10
        try:
            heuristic = _make_heuristic(prizes, distance_matrix, max_len, params)
            if not np.all(np.isfinite(heuristic)):
                return 1e10
            score = _simulate_aco(prizes, distance_matrix, max_len, heuristic, n_ants=10, n_iters=10, seed=42)
            return -score
        except Exception:
            return 1e10
    
    # Nelder-Mead optimization
    result = minimize(
        objective,
        initial_params,
        method='Nelder-Mead',
        options={'maxiter': n_evals, 'xatol': 1e-3, 'fatol': 1e-4, 'disp': False}
    )
    
    return result.x


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ACO orienteering using parameter optimization.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)
    
    start_time = time.time()
    deadline = 300.0 - 10.0  # 10 second safety margin
    
    # Initial parameters: (prize_exp, dist_exp, origin_exp, prize_w, dist_w, origin_w)
    initial_params = np.array([1.0, 2.5, 1.5, 1.0, 1.0, 1.0])
    
    # Try a few different starting points and pick the best
    best_params = initial_params
    best_score = -np.inf
    
    starting_points = [
        np.array([1.0, 2.5, 1.5, 1.0, 1.0, 1.0]),
        np.array([1.5, 2.0, 1.0, 1.0, 1.0, 1.0]),
        np.array([0.8, 3.0, 2.0, 1.0, 1.0, 1.0]),
        np.array([1.2, 2.2, 1.2, 1.5, 0.8, 1.2]),
    ]
    
    for sp in starting_points:
        if time.time() - start_time > deadline:
            break
        
        try:
            optimized_params = _optimize_parameters(
                prizes, distance_matrix, max_len, sp,
                n_evals=30, time_budget=min(20.0, deadline - (time.time() - start_time))
            )
            
            if time.time() - start_time > deadline:
                break
            
            # Evaluate the optimized parameters
            heuristic = _make_heuristic(prizes, distance_matrix, max_len, optimized_params)
            if not np.all(np.isfinite(heuristic)):
                continue
            
            score = _simulate_aco(prizes, distance_matrix, max_len, heuristic, n_ants=20, n_iters=30, seed=42)
            
            if score > best_score:
                best_score = score
                best_params = optimized_params.copy()
                
        except Exception:
            continue
    
    # If optimization failed or timed out, use a reasonable default
    if best_score == -np.inf:
        best_params = initial_params
    
    # Generate final heuristic
    heuristic = _make_heuristic(prizes, distance_matrix, max_len, best_params)
    
    # Ensure validity
    if not np.all(np.isfinite(heuristic)):
        # Fallback to simple heuristic
        d_to_origin = distance_matrix[0, :]
        prize_term = np.sqrt(prizes[None, :]) + 1e-9
        dist_penalty = 1.0 / (distance_matrix + 1e-9) ** 2.5
        origin_bonus = 1.0 / (d_to_origin[None, :] + 1e-9) ** 1.5
        heuristic = prize_term * dist_penalty * origin_bonus
    
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
