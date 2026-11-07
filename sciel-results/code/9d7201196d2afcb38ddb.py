"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _simulate_aco(prizes, distance_matrix, max_len, heuristic, seed):
    """
    Simulate the ACO colony and return the best prize collected by any ant.
    Uses full parameters: 20 ants, 30 iterations.
    """
    np.random.seed(seed)
    
    n = len(prizes)
    if n <= 1:
        return 0.0
    
    n_ants = 20
    n_iter = 30
    evaporation = 0.9
    deposition = 1.0 / 10.0
    
    # Pheromone matrix initialized to 1
    pheromone = np.ones((n, n), dtype=np.float64)
    
    # Ensure heuristic is positive
    h = np.maximum(heuristic, 1e-9)
    
    best_total = 0.0
    
    # Precompute d_j0 for speed
    d_j0 = distance_matrix[:, 0]
    
    for it in range(n_iter):
        iter_best = 0.0
        
        for ant in range(n_ants):
            visited_set = {0}
            current = 0
            length = 0.0
            total_prize = 0.0 # prizes[0] is 0
            
            while True:
                # Find eligible next nodes
                eligible = []
                for j in range(1, n):
                    if j not in visited_set:
                        d_ij = distance_matrix[current, j]
                        if length + d_ij + d_j0[j] <= max_len:
                            eligible.append(j)
                
                if not eligible:
                    break
                
                # Compute probabilities proportional to pheromone * heuristic
                probs = np.array([pheromone[current, j] * h[current, j] for j in eligible], dtype=np.float64)
                total_p = probs.sum()
                
                if total_p <= 0 or not np.isfinite(total_p):
                    break
                
                probs = probs / total_p
                probs = np.maximum(probs, 0)
                probs_sum = probs.sum()
                if probs_sum <= 0:
                    break
                probs = probs / probs_sum
                
                r = np.random.random()
                cumsum = np.cumsum(probs)
                idx = np.searchsorted(cumsum, r)
                if idx >= len(eligible):
                    idx = len(eligible) - 1
                
                next_j = eligible[idx]
                
                length += distance_matrix[current, next_j]
                total_prize += prizes[next_j]
                visited_set.add(next_j)
                current = next_j
            
            if total_prize > iter_best:
                iter_best = total_prize
        
        if iter_best > best_total:
            best_total = iter_best
        
        # Update pheromone
        pheromone = pheromone * evaporation
        pheromone += deposition * iter_best
    
    return best_total

def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness heuristic matrix for ACO orienteering.
    
    Strategy:
    1. Define a parameterized family of heuristics with regret term:
       h[i, j] = (prizes[j]^beta * exp(-d[j, 0] / scale) * (1 + regret_factor * max_prize_reachable)) / (distance_matrix[i, j] + eps)^alpha
    2. Search over a grid of (alpha, scale, beta, regret_factor) parameters.
    3. Evaluate each candidate using a full-fidelity ACO simulation (20 ants, 30 iters) with a fixed seed.
    4. Return the heuristic that yielded the best score within the time budget.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    if n == 0:
        return np.array([])
    
    if n == 1:
        return np.ones((1, 1))
    
    eps = 1e-9
    d_j0 = distance_matrix[:, 0]
    
    # Time management
    start_time = time.time()
    time_limit = 300.0
    
    best_score = -1.0
    best_h = None
    
    # Parameter ranges
    alphas = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    scales = [0.3, 0.5, 0.7, 1.0, 1.5]
    betas = [1.0, 1.2, 1.5, 2.0]
    # Regret factor: 0 (no regret), 0.5, 1.0, 1.5
    regrets = [0.0, 0.5, 1.0, 1.5]
    
    # Generate all combinations
    candidates = []
    for alpha in alphas:
        for scale in scales:
            for beta in betas:
                for regret in regrets:
                    candidates.append((alpha, scale, beta, regret))
    
    # Shuffle to avoid bias towards first parameters
    np.random.seed(42)
    indices = np.random.permutation(len(candidates))
    
    # Precompute node-level quantities that don't depend on alpha
    max_prize = np.max(prizes) if n > 0 else 0.0
    
    for idx in indices:
        if time.time() - start_time > time_limit:
            break
            
        alpha, scale, beta, regret = candidates[idx]
        
        # Compute heuristic components
        prize_pow = np.maximum(prizes, eps) ** beta
        exp_term = np.exp(-d_j0 / scale)
        
        # Node attractiveness: prize[j] * exp(-d_j0/scale)
        node_attr = prize_pow * exp_term
        
        # Add regret term: penalize nodes where the maximum reachable prize from that node is much higher
        # For simplicity, use a global regret factor that scales with the node's prize relative to max
        if regret > 0:
            # Regret: higher for nodes that are "cheap" but have high potential
            # Use 1 - (prize[j] / max_prize) as a proxy for regret
            regret_term = 1 + regret * (1 - prizes / max_prize) if max_prize > 0 else 1
            node_attr = node_attr * regret_term
        
        # Edge penalty: d[i,j]^alpha
        d_plus = distance_matrix + eps
        edge_pen = d_plus ** alpha
        
        # h[i,j] = node_attr[j] / edge_pen[i,j]
        h = node_attr[None, :] / edge_pen
        
        # Ensure positivity
        h = np.maximum(h, eps)
        
        # Evaluate with full simulation
        score = _simulate_aco(prizes, distance_matrix, max_len, h, seed=42)
        
        if score > best_score:
            best_score = score
            best_h = h.copy()
            
    # Fallback if no candidate was evaluated (should not happen)
    if best_h is None:
        d_plus = distance_matrix + eps
        node_attr = prizes * np.exp(-d_j0 / 0.5)
        best_h = node_attr[None, :] / d_plus
        best_h = np.maximum(best_h, eps)
        
    return best_h
# EVOLVE-BLOCK-END
