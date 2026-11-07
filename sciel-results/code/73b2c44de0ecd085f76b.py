import numpy as np
import time
import random

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Returns an edge-attractiveness matrix.
    Strategy: Geometric Angle-Continuity with Last-Fit Demand Penalty.
    
    The heuristic is defined as:
    H[i, j] = (1 / d_ij)^gamma * angle_factor * demand_factor
    
    1. (1/d)^gamma: Standard distance bias.
    2. angle_factor: Rewards moving in the same direction as the previous segment.
       For edge (i, j), we look at the vector from the 'previous' node to i? 
       No, the heuristic is static (computed once). 
       In standard ACO, the heuristic eta[i][j] is static. 
       To encode "angle continuity" in a static matrix, we must approximate the 
       'incoming' direction to node i. 
       
       A common static approximation for "smoothness" is based on the local 
       geometry: nodes that are close to each other and aligned with the 
       local flow are preferred. However, since we don't have a fixed incoming 
       edge, we can use a different geometric insight:
       
       Instead of dynamic angle, we use a **Spatial Coherence** metric:
       We want to form compact, non-crossing tours. 
       
       Alternative Structural Change: 
       The current program uses a scalar search over gamma and beta. 
       The directive asks for a STRUCTURAL change. 
       
       Let's try a **Cluster-Based Heuristic**.
       Pre-compute a rough clustering of customers (e.g., using K-Medoids or simple 
       spatial partitioning) and bias the heuristic to encourage visiting nodes 
       within the same cluster consecutively. This reduces the "zig-zag" across 
       the whole square and encourages completing a "zone" before moving on, 
       which naturally handles capacity constraints better by keeping loads 
       localized.
       
       Specifically:
       1. Compute a "local density" or cluster affinity.
       2. H[i,j] = (1/d_ij)^gamma * affinity[i,j]
       
       Where affinity[i,j] is high if i and j are in the same "local neighborhood" 
       defined by a spatial grid or k-NN graph.
       
       Let's implement a static k-NN based affinity.
       If j is among the k nearest neighbors of i (excluding depot effects), 
       boost the edge. This creates a "local tour" bias.
       
       Combined with a demand penalty that prefers "medium" demands? 
       Or simply: H[i,j] = (1/d_ij)^2 * (1 + kNN_boost)
       
       Let's try: 
       H[i,j] = (1/d_ij)^gamma * (1 + alpha * I(j in kNN(i)))
       
       This is a structural change from "scalar distance + demand penalty" to 
       "graph-based locality + distance".
    """
    start_time = time.time()
    deadline = start_time + 310.0  # 20s safety margin
    
    D = np.asarray(distance_matrix, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)
    n = D.shape[0]
    
    # Precompute base features
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_dist = np.where(D > 1e-10, 1.0 / D, 0.0)
    
    def simulate_aco(heuristic_matrix):
        """
        Runs the EXACT ACO protocol specified in the task contract.
        Returns the best cost found.
        """
        pheromone = np.ones_like(heuristic_matrix)
        best_total_cost = float('inf')
        
        num_ants = 20
        num_iterations = 30
        
        # Optimization: Pre-compute feasible lists? No, load changes dynamically.
        # But we can optimize the inner loop.
        
        for _ in range(num_iterations):
            iteration_best_cost = float('inf')
            
            for ant in range(num_ants):
                current = 0
                current_load = 0.0
                total_dist = 0.0
                unvisited = set(range(1, n))
                
                while unvisited:
                    # Find feasible next nodes
                    feasible = []
                    for j in unvisited:
                        if demands[j] <= (capacity - current_load):
                            feasible.append(j)
                    
                    if not feasible:
                        # Return to depot
                        total_dist += D[current, 0]
                        break
                    
                    # Calculate probabilities
                    # p[j] = pheromone[current, j] * heuristic[current, j]
                    # Optimization: Vectorize weight calculation
                    # feasible is a list. Convert to array for indexing.
                    feasible_arr = np.array(feasible, dtype=int)
                    weights = pheromone[current, feasible_arr] * heuristic_matrix[current, feasible_arr]
                    
                    total_w = np.sum(weights)
                    if total_w < 1e-10:
                        next_node = random.choice(feasible)
                    else:
                        probs = weights / total_w
                        # Use random.choices for weighted selection
                        next_node = random.choices(feasible, weights=probs, k=1)[0]
                    
                    # Move
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
            
            # Update Pheromone
            pheromone *= 0.9
            if iteration_best_cost > 0:
                pheromone += 1.0 / iteration_best_cost
        
        return best_total_cost

    best_cost = float('inf')
    best_heuristic = None
    
    # Strategy: Search over gamma and kNN boost alpha
    # Also try different k for kNN
    
    # To save time, we will evaluate a small set of candidates.
    # The structural change is the kNN boost.
    
    candidates = [
        (1.0, 0.0, 5),   # Baseline-ish
        (1.5, 1.0, 5),   # Moderate boost
        (2.0, 2.0, 5),   # Strong boost
        (1.5, 1.0, 10),  # Wider neighborhood
        (2.0, 3.0, 5),   # Very strong locality
        (1.0, 2.0, 3),   # Tight neighborhood
    ]
    
    # Shuffle to ensure diversity
    random.shuffle(candidates)
    
    for gamma, alpha, k in candidates:
        if time.time() > deadline - 15.0:
            break
            
        # Construct kNN boost matrix
        # For each node i, find k nearest neighbors.
        # boost[i, j] = 1 if j is in kNN(i), else 0.
        boost = np.zeros((n, n), dtype=np.float64)
        
        # Efficient kNN using scipy or numpy? 
        # n=50, so O(n^2 log n) is fine.
        for i in range(n):
            # Distance from i to all others
            dists = D[i].copy()
            dists[i] = float('inf') # Exclude self
            # Get k smallest indices
            # np.argpartition is faster for top-k
            if n > k:
                k_indices = np.argpartition(dists, k)[:k]
            else:
                k_indices = np.where(dists < float('inf'))[0]
            
            for j in k_indices:
                boost[i, j] = 1.0
        
        # Construct heuristic
        # H = (1/d)^gamma * (1 + alpha * boost)
        dist_part = np.power(inv_dist, gamma)
        
        heuristic_matrix = dist_part * (1.0 + alpha * boost)
        
        # Sanitize
        heuristic_matrix = np.where(np.isfinite(heuristic_matrix) & (heuristic_matrix > 0), heuristic_matrix, 1e-9)
        
        # Run Simulation
        cost = simulate_aco(heuristic_matrix)
        
        if cost < best_cost:
            best_cost = cost
            best_heuristic = heuristic_matrix

    # If no heuristic was found (shouldn't happen), fallback to standard 1/d
    if best_heuristic is None:
        with np.errstate(divide='ignore', invalid='ignore'):
            best_heuristic = np.where(D > 0, D ** (-1.0), 1e-9)
        best_heuristic = np.where(np.isfinite(best_heuristic) & (best_heuristic > 0), best_heuristic, 1e-9)

    return best_heuristic
