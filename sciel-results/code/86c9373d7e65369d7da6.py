# EVOLVE-BLOCK-START
import numpy as np
import time


def _proxy_aco(distance_matrix, coordinates, demands, capacity, heuristic, n_ants=20, n_iters=30, seed=42):
    """Run a proxy ACO to evaluate a heuristic's quality."""
    rng = np.random.default_rng(seed)
    n = distance_matrix.shape[0]
    dm = distance_matrix
    eta = heuristic
    tau = np.ones((n, n))
    best_cost = np.inf
    
    for it in range(n_iters):
        iter_best = np.inf
        for _ in range(n_ants):
            current = 0
            load = 0.0
            visited = set()
            route_cost = 0.0
            while True:
                # Find feasible unvisited customers
                feasible = []
                for j in range(1, n):
                    if j not in visited and demands[j] <= capacity - load:
                        feasible.append(j)
                if not feasible:
                    # Return to depot
                    route_cost += dm[current, 0]
                    break
                # Compute probabilities
                probs = np.zeros(len(feasible))
                for idx, j in enumerate(feasible):
                    probs[idx] = tau[current, j] * eta[current, j]
                total = probs.sum()
                if total <= 0:
                    probs = np.ones(len(feasible))
                    total = probs.sum()
                probs /= total
                # Select next node
                r = rng.random()
                cumsum = np.cumsum(probs)
                idx = np.searchsorted(cumsum, r)
                if idx >= len(feasible):
                    idx = len(feasible) - 1
                next_node = feasible[idx]
                route_cost += dm[current, next_node]
                load += demands[next_node]
                visited.add(next_node)
                current = next_node
            if route_cost < iter_best:
                iter_best = route_cost
        if iter_best < best_cost:
            best_cost = iter_best
        # Update pheromone
        tau *= 0.9
        tau += 1.0 / max(iter_best, 1e-9)
    
    return best_cost


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute heuristic matrix for ACO using a combination of distance-based and demand-based heuristics.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    dm = np.asarray(distance_matrix, dtype=np.float64)
    n = dm.shape[0]
    
    # Base heuristic: 1 / (d + eps)^gamma
    # Try a range of gamma values
    gammas = np.array([0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0])
    
    best_gamma = 1.0
    best_score = np.inf
    
    for gamma in gammas:
        if time.time() > deadline:
            break
        eta = 1.0 / (dm + 1e-9) ** gamma
        # Ensure positive values
        eta[~np.isfinite(eta)] = 1e-9
        eta[eta <= 0] = 1e-9
        
        # Run proxy ACO with this heuristic
        score = _proxy_aco(dm, coordinates, demands, capacity, eta, n_ants=20, n_iters=30, seed=42)
        
        if score < best_score:
            best_score = score
            best_gamma = gamma
    
    # Final heuristic with best gamma
    eta = 1.0 / (dm + 1e-9) ** best_gamma
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    return eta
# EVOLVE-BLOCK-END
