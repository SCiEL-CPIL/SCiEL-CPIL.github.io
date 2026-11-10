"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _fast_aco_eval(distance_matrix, demands, capacity, heuristic, n_ants=5, n_iters=5, seed=42):
    """
    A very fast, simplified ACO evaluation to approximate the true ACO performance.
    This is used to filter the top candidates from the greedy pre-screening.
    """
    n = len(demands)
    rng = np.random.RandomState(seed)
    pheromone = np.ones_like(distance_matrix)
    best_overall = np.inf
    
    for it in range(n_iters):
        best_iter_cost = np.inf
        
        for ant in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            remaining_cap = capacity
            total_dist = 0.0
            
            while not np.all(visited):
                scores = pheromone[current] * heuristic[current]
                
                # Mask visited and infeasible
                feasible_mask = (~visited) & (demands <= remaining_cap)
                scores[~feasible_mask] = 0.0
                scores[0] = 0.0
                
                if not np.any(scores > 0):
                    total_dist += distance_matrix[current, 0]
                    current = 0
                    remaining_cap = capacity
                    continue
                
                total_score = np.sum(scores)
                if total_score <= 0:
                    feasible_indices = np.where(feasible_mask)[0]
                    if len(feasible_indices) == 0:
                        total_dist += distance_matrix[current, 0]
                        current = 0
                        remaining_cap = capacity
                        continue
                    next_node = feasible_indices[rng.randint(len(feasible_indices))]
                else:
                    probs = scores / total_score
                    next_node = rng.choice(n, p=probs)
                
                total_dist += distance_matrix[current, next_node]
                remaining_cap -= demands[next_node]
                visited[next_node] = True
                current = next_node
            
            total_dist += distance_matrix[current, 0]
            
            if total_dist < best_iter_cost:
                best_iter_cost = total_dist
        
        # Update pheromone
        pheromone *= 0.9
        pheromone += 1.0 / best_iter_cost
        
        if best_iter_cost < best_overall:
            best_overall = best_iter_cost
    
    return best_overall

def _greedy_eval_fast(distance_matrix, demands, capacity, heuristic):
    """
    Fast greedy evaluation for initial screening.
    """
    n = len(demands)
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    current = 0
    remaining_cap = capacity
    total_dist = 0.0
    
    for _ in range(n - 1):
        scores = heuristic[current].copy()
        feasible_mask = (~visited) & (demands <= remaining_cap)
        scores[~feasible_mask] = -np.inf
        scores[0] = -np.inf
        
        if not np.any(np.isfinite(scores)):
            total_dist += distance_matrix[current, 0]
            current = 0
            remaining_cap = capacity
            continue
        
        best_j = np.argmax(scores)
        total_dist += distance_matrix[current, best_j]
        remaining_cap -= demands[best_j]
        visited[best_j] = True
        current = best_j
    
    total_dist += distance_matrix[current, 0]
    return total_dist

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Returns an edge-attractiveness matrix using a node-potential field model.
    
    The heuristic is structured as:
    H[i,j] = (1/d[i,j]) * load_balance_pressure[j]
    
    Where load_balance_pressure[j] is derived from the spatial density of unvisited
    demand relative to the centroid, guiding ants toward regions that maximize
    local demand aggregation.
    """
    n = len(demands)
    d_safe = distance_matrix + 1e-9
    start_time = time.time()
    deadline = start_time + 295.0  # Safety margin
    
    # Customer nodes are 1..n-1
    customer_indices = np.arange(1, n)
    customer_coords = coordinates[1:]
    customer_demands = demands[1:]
    
    # Compute centroid of customers (weighted by demand)
    total_demand = np.sum(customer_demands)
    if total_demand > 0:
        centroid = np.average(customer_coords, weights=customer_demands, axis=0)
    else:
        centroid = np.mean(customer_coords, axis=0)
    
    # Compute distance from each customer to centroid
    dist_to_centroid = np.linalg.norm(customer_coords - centroid, axis=1)
    
    # Node potential: higher for nodes closer to centroid (dense demand regions)
    # and for nodes with higher demand
    # potential[j] = demand[j] * exp(-beta * dist_to_centroid[j])
    
    def make_node_potential_heuristic(alpha, beta, gamma):
        """
        alpha: distance decay exponent for 1/d[i,j]
        beta: spatial decay rate for centroid distance
        gamma: demand weighting strength
        """
        # Node potential based on demand and proximity to centroid
        spatial_weight = np.exp(-beta * dist_to_centroid)
        demand_weight = 1.0 + gamma * customer_demands
        node_potential = demand_weight * spatial_weight
        
        # Distance component: closer is better
        dist_component = 1.0 / (d_safe ** alpha)
        
        # Build full heuristic matrix
        # H[i,j] = dist_component[i,j] * node_potential[j] for j >= 1
        # For j = 0 (depot), use a neutral value
        h = np.zeros((n, n))
        
        # For customer-to-customer edges
        h[1:, 1:] = dist_component[1:, 1:] * node_potential[None, :]
        
        # For depot-to-customer edges
        h[0, 1:] = dist_component[0, 1:] * node_potential
        
        # For customer-to-depot edges
        h[1:, 0] = dist_component[1:, 0]
        
        # For depot-to-depot
        h[0, 0] = 1e-9
        
        # Ensure positivity and valid shape
        h = np.maximum(h, 1e-9)
        np.fill_diagonal(h, 1e-9)
        
        return h

    # Generate diverse candidates
    candidate_params = []
    
    # Systematic exploration
    for alpha in [0.8, 1.0, 1.2]:
        for beta in [0.5, 1.0, 2.0, 3.0]:
            for gamma in [0.5, 1.0, 2.0]:
                candidate_params.append((alpha, beta, gamma))
    
    # Random sampling
    rng = np.random.RandomState(42)
    for _ in range(20):
        a = rng.uniform(0.5, 1.5)
        b = rng.uniform(0.3, 4.0)
        g = rng.uniform(0.3, 3.0)
        candidate_params.append((a, b, g))
    
    # Phase 1: Greedy Screening
    heuristics_cache = []
    
    for a, b, g in candidate_params:
        if time.time() > deadline - 30.0:
            break
        h = make_node_potential_heuristic(a, b, g)
        score = _greedy_eval_fast(distance_matrix, demands, capacity, h)
        heuristics_cache.append((score, h, a, b, g))
    
    # Sort by greedy score
    heuristics_cache.sort(key=lambda x: x[0])
    
    # Phase 2: Fast ACO Verification
    top_k = min(6, len(heuristics_cache))
    top_candidates = heuristics_cache[:top_k]
    
    best_aco_score = np.inf
    best_heuristic = top_candidates[0][1] if top_candidates else make_node_potential_heuristic(1.0, 1.0, 1.0)
    
    for score, h, a, b, g in top_candidates:
        if time.time() > deadline - 10.0:
            break
            
        aco_score = _fast_aco_eval(distance_matrix, demands, capacity, h)
        
        if aco_score < best_aco_score:
            best_aco_score = aco_score
            best_heuristic = h

    # Final Safety Checks
    if not np.all(np.isfinite(best_heuristic)):
        return 1.0 / d_safe
    if np.any(best_heuristic <= 0):
        best_heuristic = np.clip(best_heuristic, 1e-9, None)
    if best_heuristic.shape != distance_matrix.shape:
        return 1.0 / d_safe
        
    return best_heuristic
# EVOLVE-BLOCK-END
