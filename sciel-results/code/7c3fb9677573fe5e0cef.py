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
    
    # Precompute log probabilities to avoid overflow/underflow in exp? 
    # Actually, just normalizing is fine for small n.
    
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
                    # Fallback
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
    Returns an edge-attractiveness matrix.
    Strategy: 
    1. Define a simple parametric family: H[i,j] = 1 / (d[i,j]^alpha * (1 + beta * d[j]/C))
    2. Generate a diverse set of candidates.
    3. Screen them using fast greedy evaluation.
    4. Select the top K candidates.
    5. Evaluate top K using a fast ACO simulation.
    6. Return the heuristic with the best fast ACO score.
    """
    n = len(demands)
    d_safe = distance_matrix + 1e-9
    start_time = time.time()
    deadline = start_time + 300.0  # Safety margin
    
    # Precompute demand ratio
    demand_ratio = demands / capacity
    
    def make_heuristic(alpha, beta):
        # H[i,j] = 1 / (d[i,j]^alpha * (1 + beta * demand[j]/C))
        # Note: demand_ratio is 1D, so we broadcast to columns (j)
        denom = (d_safe ** alpha) * (1.0 + beta * demand_ratio[None, :])
        h = 1.0 / denom
        h = np.maximum(h, 1e-9)
        np.fill_diagonal(h, 1e-9)
        return h

    # Generate candidates
    # Alpha: distance exponent. Higher alpha makes the heuristic more "greedy" (closer is much better).
    # Beta: demand penalty. Higher beta penalizes high-demand customers more.
    
    candidate_params = []
    
    # Base standard
    candidate_params.append((1.0, 0.0))
    
    # Distance exponents
    for alpha in [0.8, 1.2, 1.5, 2.0]:
        candidate_params.append((alpha, 0.0))
        
    # Demand penalties
    for beta in [0.5, 1.0, 2.0]:
        candidate_params.append((1.0, beta))
        
    # Combinations
    for alpha in [1.2, 1.5]:
        for beta in [0.5, 1.0]:
            candidate_params.append((alpha, beta))
            
    # Random sampling around the best known good regions
    rng = np.random.RandomState(42)
    for _ in range(20):
        a = rng.uniform(0.5, 2.5)
        b = rng.uniform(0.0, 2.5)
        candidate_params.append((a, b))

    # Phase 1: Greedy Screening
    # Evaluate all candidates with fast greedy
    greedy_scores = []
    heuristics_cache = []
    
    for a, b in candidate_params:
        if time.time() > deadline - 30.0:
            break
        h = make_heuristic(a, b)
        score = _greedy_eval_fast(distance_matrix, demands, capacity, h)
        greedy_scores.append(score)
        heuristics_cache.append((score, h, a, b))
        
    # Sort by greedy score
    heuristics_cache.sort(key=lambda x: x[0])
    
    # Phase 2: Fast ACO Verification
    # Take the top 5 candidates
    top_k = 5
    top_candidates = heuristics_cache[:top_k]
    
    best_aco_score = np.inf
    best_heuristic = top_candidates[0][1] # Fallback to best greedy
    
    for score, h, a, b in top_candidates:
        if time.time() > deadline - 10.0:
            break
            
        # Run fast ACO
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
