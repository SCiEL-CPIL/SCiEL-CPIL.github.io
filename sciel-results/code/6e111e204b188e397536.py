# EVOLVE-BLOCK-START
import numpy as np
import time

def _construct_tour(distance_matrix, heuristic_matrix, demands, capacity):
    """
    Constructs a tour using a greedy strategy driven by the heuristic matrix.
    The heuristic matrix H[i,j] is interpreted as an attractiveness score.
    We pick the next node j maximizing H[current, j] among feasible nodes.
    If no feasible node exists, return to depot.
    """
    n = len(demands)
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    current = 0
    remaining_capacity = capacity
    total_distance = 0.0
    
    unvisited_count = n - 1
    while unvisited_count > 0:
        best_j = -1
        best_score = -np.inf
        
        # Vectorized search for best next node
        # Mask out visited and infeasible nodes
        feasible_mask = ~visited
        feasible_mask[0] = False # Never go to depot as a customer
        # Check capacity
        # demands is a numpy array
        feasible_mask[1:] = feasible_mask[1:] & (demands[1:] <= remaining_capacity)
        
        if not np.any(feasible_mask):
            # Return to depot
            total_distance += distance_matrix[current, 0]
            current = 0
            remaining_capacity = capacity
            continue
            
        # Get scores for feasible nodes
        scores = heuristic_matrix[current].copy()
        scores[~feasible_mask] = -np.inf
        
        best_j = np.argmax(scores)
        best_score = scores[best_j]
        
        if best_score == -np.inf:
            # Should be caught by the mask check above, but safety
            total_distance += distance_matrix[current, 0]
            current = 0
            remaining_capacity = capacity
            continue

        # Move to best_j
        total_distance += distance_matrix[current, best_j]
        remaining_capacity -= demands[best_j]
        visited[best_j] = True
        unvisited_count -= 1
        current = best_j
        
    # Return to depot at the end
    total_distance += distance_matrix[current, 0]
    
    return total_distance

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Generates an edge-attractiveness matrix.
    Strategy: Use the heuristic matrix to guide a constructive heuristic.
    We will optimize the heuristic parameters to minimize the cost of the
    tour constructed by the greedy algorithm described in _construct_tour.
    
    The heuristic form is: H[i,j] = 1 / (distance[i,j] * (1 + distance[i,0] * alpha + distance[j,0] * beta))
    Or simpler: H[i,j] = 1 / distance[i,j]^p * (1 + depot_factor)
    
    Let's try a parameterized family:
    H[i,j] = 1 / (distance[i,j] ** p) * (1 / (1 + gamma * distance[0,j]))
    
    We will search over p and gamma.
    """
    n = distance_matrix.shape[0]
    start_time = time.time()
    deadline = start_time + 290.0 # Safety margin for 300s limit
    
    # Pre-compute safe distances
    eps = 1e-9
    safe_dist = distance_matrix + eps
    depot_dist = distance_matrix[0, :]
    
    # Candidate parameters
    # p: exponent for distance (1.0 is standard, >1 makes it sharper)
    # gamma: depot bias (0 is no bias, >1 favors nodes close to depot)
    
    p_candidates = [1.0, 1.2, 1.5, 2.0]
    gamma_candidates = [0.0, 0.5, 1.0, 2.0, 5.0]
    
    best_h = None
    best_cost = float('inf')
    
    # We can also try a "cluster" heuristic:
    # H[i,j] is high if i and j are in the same "region" (close to each other)
    # and low if they are far. This is essentially 1/d^p.
    # The depot bias helps ensure that the tour doesn't get "stuck" far from depot
    # with high remaining capacity, or forces early returns if capacity is low?
    # Actually, in ACO, the heuristic is static. The capacity constraint is handled
    # by the feasibility check in the ant's move.
    # A good heuristic should encourage short edges.
    # A "depot-aware" heuristic might help in balancing the routes?
    
    # Let's evaluate a grid of (p, gamma)
    for p in p_candidates:
        if time.time() > deadline:
            break
        for gamma in gamma_candidates:
            if time.time() > deadline:
                break
            
            # Construct heuristic
            # H[i,j] = 1 / (d_ij^p) * (1 / (1 + gamma * d_0j))
            # Note: d_0j is the distance from depot to j.
            # This makes edges to nodes close to depot more attractive.
            
            # Optimization: Pre-compute 1/d^p
            # But p changes.
            
            # Vectorized construction
            # d_pow = safe_dist ** p
            # inv_d_pow = 1.0 / d_pow
            # depot_scale = 1.0 / (1.0 + gamma * depot_dist)
            # H = inv_d_pow * depot_scale[np.newaxis, :]
            
            # To save time, we can compute 1/d^p for each p once.
            # Let's just do it inside the loop, it's fast enough for 50x50.
            
            try:
                d_pow = safe_dist ** p
                inv_d_pow = 1.0 / d_pow
                depot_scale = 1.0 / (1.0 + gamma * depot_dist)
                h = inv_d_pow * depot_scale[np.newaxis, :]
                
                # Ensure finite and positive
                if not np.all(np.isfinite(h)):
                    continue
                h = np.where(h > 0, h, 1e-9)
                
                cost = _construct_tour(distance_matrix, h, demands, capacity)
                
                if cost < best_cost:
                    best_cost = cost
                    best_h = h.copy()
                    
            except Exception:
                continue
                
    # If no valid heuristic was found (should not happen), fallback to 1/d
    if best_h is None:
        best_h = 1.0 / safe_dist
        best_h = np.where(best_h > 0, best_h, 1e-9)
        
    # Final sanity check
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones_like(distance_matrix) * 1e-9
        
    return best_h
# EVOLVE-BLOCK-END
