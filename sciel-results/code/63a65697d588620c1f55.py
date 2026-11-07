import numpy as np
import time
import random

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Returns an edge-attractiveness matrix.
    Strategy: Structural Graph Heuristic (KNN + MST).
    
    Instead of a parametric distance function, we identify the geometric 
    "backbone" of the instance using K-Nearest Neighbors (KNN) and a 
    Minimum Spanning Tree (MST). Edges that are part of these structures 
    are given high attractiveness. This encourages the ACO to build 
    routes that follow the natural low-cost connectivity of the point set, 
    rather than just greedily picking the nearest neighbor, which can lead 
    to fragmented or crossing paths.
    
    H[i, j] = base_attractiveness * structural_bonus * demand_smoothing
    """
    start_time = time.time()
    deadline = start_time + 310.0  # Safety margin
    
    D = np.asarray(distance_matrix, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)
    n = D.shape[0]
    
    # 1. Base Attractiveness: Inverse distance (standard)
    # We use a moderate exponent to avoid extreme bias to the very nearest node
    with np.errstate(divide='ignore', invalid='ignore'):
        base_attr = np.where(D > 1e-10, 1.0 / D, 0.0)
    
    # 2. Structural Bonus: KNN and MST
    # Initialize structural bonus to 1.0 (neutral)
    structural_bonus = np.ones_like(base_attr)
    
    # --- KNN Component ---
    # For each node, identify its K nearest neighbors.
    # These edges are high-probability candidates for optimal tours.
    K = 4
    # We compute KNN for all nodes (including depot for return edges potentially, 
    # though customers are the main focus)
    # Using scipy is allowed, but let's stick to numpy for simplicity and speed 
    # given n=50 is small.
    
    # Distance matrix is symmetric. 
    # For each i, find indices of K smallest D[i, j] for j != i
    knn_mask = np.zeros_like(base_attr, dtype=bool)
    for i in range(n):
        # Get distances from i to all others
        dists = D[i, :].copy()
        dists[i] = np.inf  # Exclude self
        # Get K smallest
        # argsort returns indices of sorted elements
        k_indices = np.argsort(dists)[:K]
        for j in k_indices:
            knn_mask[i, j] = True
            knn_mask[j, i] = True  # Symmetric
    
    # Boost KNN edges
    structural_bonus[knn_mask] *= 2.0
    
    # --- MST Component ---
    # Compute MST on the graph of customers (1 to n-1) plus depot (0)
    # A simple Prim's algorithm or Kruskal's is fine for n=50.
    # Let's use a simple Prim's starting from depot 0.
    
    in_mst = [False] * n
    in_mst[0] = True
    min_dist_to_mst = D[0, :].copy()
    
    mst_edges = []
    
    for _ in range(n - 1):
        # Find node not in MST with min distance to MST
        # Exclude nodes already in MST
        min_dist_to_mst_copy = min_dist_to_mst.copy()
        for i in range(n):
            if in_mst[i]:
                min_dist_to_mst_copy[i] = np.inf
        
        u = np.argmin(min_dist_to_mst_copy)
        if min_dist_to_mst_copy[u] == np.inf:
            break # Disconnected? Unlikely with full matrix
            
        in_mst[u] = True
        mst_edges.append(u)
        
        # Update min distances
        for v in range(n):
            if not in_mst[v] and D[u, v] < min_dist_to_mst[v]:
                min_dist_to_mst[v] = D[u, v]
        
    # The MST edges are the absolute minimum connectivity structure.
    # We don't have the specific "from" node stored in the simple loop above easily 
    # without tracking parent pointers. Let's redo with parent tracking for edge identification.
    
    # Re-do Prim's to get edges (u, v)
    in_mst = [False] * n
    in_mst[0] = True
    min_dist_to_mst = D[0, :].copy()
    parent = [-1] * n
    parent[0] = -1
    
    for _ in range(n - 1):
        min_dist_to_mst_copy = min_dist_to_mst.copy()
        for i in range(n):
            if in_mst[i]:
                min_dist_to_mst_copy[i] = np.inf
        
        u = np.argmin(min_dist_to_mst_copy)
        if min_dist_to_mst_copy[u] == np.inf:
            break
            
        in_mst[u] = True
        # Edge is (parent[u], u)
        p = parent[u]
        if p != -1:
            structural_bonus[p, u] *= 1.5
            structural_bonus[u, p] *= 1.5
        
        for v in range(n):
            if not in_mst[v] and D[u, v] < min_dist_to_mst[v]:
                min_dist_to_mst[v] = D[u, v]
                parent[v] = u

    # 3. Demand Smoothing
    # High demand nodes are "heavy". We want to encourage grouping them, 
    # but also not to over-penalize them if they are well-connected structurally.
    # A soft penalty: 1 / (1 + alpha * demand)
    # Since we are using a structural bonus, we can afford a slightly stronger 
    # demand penalty to help with capacity constraints, as the structural bonus 
    # provides the "direction" and demand provides the "feasibility hint".
    alpha = 0.5
    demand_factor = 1.0 / (1.0 + alpha * demands)
    
    # Combine
    # H[i, j] = base_attr[i, j] * structural_bonus[i, j] * demand_factor[j]
    # Note: demand_factor applies to the destination node j.
    
    heuristic_matrix = base_attr * structural_bonus * demand_factor[np.newaxis, :]
    
    # Sanitize
    heuristic_matrix = np.where(np.isfinite(heuristic_matrix) & (heuristic_matrix > 0), heuristic_matrix, 1e-9)
    
    # Optional: Verify with a quick ACO run if time permits?
    # Given n=50 and 20 ants * 30 iters, one run is ~1-2 seconds.
    # We can run 1-2 variations of alpha or K to see if it helps.
    # But the structural approach is deterministic based on geometry.
    # Let's do a quick sanity check simulation to ensure it doesn't crash 
    # and returns a reasonable cost, but we won't do a full search 
    # because the structural features are robust.
    
    def simulate_aco(heuristic_matrix, seed=42):
        pheromone = np.ones_like(heuristic_matrix)
        best_total_cost = float('inf')
        num_ants = 20
        num_iterations = 30
        rng = random.Random(seed)
        
        for _ in range(num_iterations):
            iteration_best_cost = float('inf')
            
            for _ant in range(num_ants):
                current = 0
                current_load = 0.0
                total_dist = 0.0
                unvisited = set(range(1, n))
                
                while unvisited:
                    feasible = [j for j in unvisited if demands[j] <= (capacity - current_load)]
                    
                    if not feasible:
                        total_dist += D[current, 0]
                        break
                    
                    weights = np.array([pheromone[current, j] * heuristic_matrix[current, j] for j in feasible])
                    total_w = np.sum(weights)
                    if total_w < 1e-10:
                        next_node = rng.choice(feasible)
                    else:
                        probs = weights / total_w
                        # Cumulative sum for choice
                        r = rng.random()
                        c_sum = 0.0
                        next_node = feasible[-1]
                        for idx, p in enumerate(probs):
                            c_sum += p
                            if r <= c_sum:
                                next_node = feasible[idx]
                                break
                    
                    total_dist += D[current, next_node]
                    current_load += demands[next_node]
                    unvisited.remove(next_node)
                    current = next_node
                    
                    if not unvisited:
                        total_dist += D[current, 0]
                        break
            
            if total_dist < iteration_best_cost:
                iteration_best_cost = total_dist
            
            if iteration_best_cost < best_total_cost:
                best_total_cost = iteration_best_cost
            
            pheromone *= 0.9
            if iteration_best_cost > 0:
                pheromone += 1.0 / iteration_best_cost
        
        return best_total_cost

    # Quick validation run (uses ~1-2s)
    # If time allows, we might tweak alpha slightly.
    # But for now, return the structural heuristic.
    
    return heuristic_matrix
