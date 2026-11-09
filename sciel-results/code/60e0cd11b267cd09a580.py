import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes the heuristic matrix for the ant colony optimization.
    
    Uses a parametric grid search over distance and prize exponents to find
    a robust heuristic that captures 'effective cost' dynamics.
    
    The heuristic is: eta_ij = (prize_j / (d_ij * d_j0)) ^ alpha * prize_j^beta
    or similar parametric forms, searched over a grid of parameters.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = distance_matrix.shape[0]
    if n < 2:
        return np.ones((n, n), dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 320.0  # safety margin before 330s limit
    
    # Precompute useful arrays
    d = distance_matrix.copy()
    d_to_depot = d[:, 0].copy()
    
    # Avoid division by zero
    eps = 1e-9
    d_safe = d + eps
    d_to_depot_safe = d_to_depot + eps
    
    # Base components
    # prize_j is the prize at node j
    # d_ij is distance from i to j
    # d_j0 is distance from j to depot
    
    # Try a grid of parametric forms
    # Form: eta_ij = prize_j^a / (d_ij^b * d_j0^c)
    # with a, b, c in some range
    
    best_score = -1
    best_eta = None
    
    # Grid search over exponents
    a_vals = [0.5, 1.0, 1.5, 2.0]
    b_vals = [0.5, 1.0, 1.5, 2.0]
    c_vals = [0.5, 1.0, 1.5, 2.0]
    
    # Also try with prize in numerator differently
    # Form 2: eta_ij = (prize_j / d_ij) * (1/d_j0)^c
    # Form 3: eta_ij = prize_j / (d_ij^b + d_j0^c)
    
    # For evaluation, we need to simulate the ACO
    # But that's expensive. Instead, use a proxy metric:
    # For each candidate heuristic, compute how well it guides ants
    # toward high-prize, short routes.
    
    # A simpler proxy: for each node j (j != 0), compute the "quality"
    # of visiting j: prize_j / (d_j0 + min_i d_ij) or similar
    # Then check if the heuristic correlates with this quality.
    
    # Actually, let's do a lightweight ACO simulation for a few iterations
    # to evaluate each candidate heuristic.
    
    def evaluate_heuristic(eta, n_ants=5, n_iter=5):
        """Quick ACO simulation to estimate heuristic quality."""
        if time.time() > deadline:
            return -1
        
        pheromone = np.ones((n, n))
        best_total = 0
        
        for _ in range(n_iter):
            iter_best = 0
            for _ in range(n_ants):
                visited = [0]
                current = 0
                length = 0.0
                total_prize = prizes[0]
                
                while True:
                    # Find candidate nodes
                    candidates = []
                    for j in range(1, n):
                        if j in visited:
                            continue
                        new_length = length + d[current, j] + d[j, 0]
                        if new_length <= max_len:
                            candidates.append(j)
                    
                    if not candidates:
                        break
                    
                    # Compute probabilities
                    probs = np.zeros(len(candidates))
                    for idx, j in enumerate(candidates):
                        probs[idx] = pheromone[current, j] * eta[current, j]
                    
                    total_prob = probs.sum()
                    if total_prob < 1e-15:
                        probs = np.ones(len(candidates))
                        total_prob = len(candidates)
                    else:
                        probs /= total_prob
                    
                    # Sample
                    r = np.random.random()
                    cumsum = 0
                    chosen_idx = len(candidates) - 1
                    for idx in range(len(candidates)):
                        cumsum += probs[idx]
                        if r <= cumsum:
                            chosen_idx = idx
                            break
                    
                    j = candidates[chosen_idx]
                    length += d[current, j]
                    total_prize += prizes[j]
                    visited.append(j)
                    current = j
                
                if total_prize > iter_best:
                    iter_best = total_prize
                
                # Update pheromone
                pheromone *= 0.9
                pheromone += iter_best / 10.0
            
            best_total = max(best_total, iter_best)
        
        return best_total
    
    # Generate candidate heuristics
    candidates = []
    
    for a in a_vals:
        for b in b_vals:
            for c in c_vals:
                # Form 1: prize_j^a / (d_ij^b * d_j0^c)
                node_part = (prizes + eps) ** a / (d_to_depot_safe ** c)
                eta = (node_part[None, :] / (d_safe ** b))
                eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
                candidates.append(eta)
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Also try the current formula as a baseline
    node_attraction = prizes / d_to_depot_safe
    eta_current = node_attraction[None, :] / d_safe
    eta_current = np.where(np.isfinite(eta_current) & (eta_current > 0), eta_current, 1e-9)
    candidates.append(eta_current)
    
    # Evaluate each candidate
    for idx, eta in enumerate(candidates):
        if time.time() > deadline:
            break
        
        score = evaluate_heuristic(eta)
        if score > best_score:
            best_score = score
            best_eta = eta
    
    # If no valid candidate was found, return the current formula
    if best_eta is None:
        best_eta = eta_current
    
    return best_eta
