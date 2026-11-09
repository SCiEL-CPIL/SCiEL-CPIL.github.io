"""Lagrangian Relaxation-based edge weight heuristic for CVRP ant colony."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix using a Lagrangian Relaxation approach.
    
    This implementation searches for an optimal Lagrange multiplier lambda that
    penalizes node demand in the edge costs. The optimal lambda is found by
    minimizing the total distance of a set of routes constructed via a greedy
    nearest-neighbor tour on the modified cost matrix, followed by capacity-based
    splitting.
    
    The heuristic matrix H is then defined as the inverse of the modified cost
    matrix using the optimal lambda.
    """
    n = distance_matrix.shape[0]
    demands = np.asarray(demands, dtype=float)
    demands[0] = 0.0  # Depot has no demand
    
    # Helper to compute total distance of routes for a given lambda
    def eval_lambda(lam):
        # Modified cost matrix: W_ij = D_ij + lam * d_j
        # Note: d_0 = 0, so W_i0 = D_i0
        # We add lam * demands[j] to column j
        W = distance_matrix + lam * demands
        # Ensure W is symmetric? No, W_ij depends on j's demand.
        # The problem is directed in terms of cost if we use this, but distance is symmetric.
        # Usually, we consider the cost of moving i->j.
        
        # Construct Nearest Neighbor Tour using W
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        current = 0
        tour = [0]
        
        for _ in range(n - 1):
            # Find nearest unvisited node
            # Mask visited nodes with inf
            W_masked = W[current].copy()
            W_masked[visited] = np.inf
            next_node = np.argmin(W_masked)
            if next_node == current:
                break
            tour.append(next_node)
            visited[next_node] = True
            current = next_node
        
        # Split the tour into routes respecting capacity
        total_dist = 0.0
        current_load = 0.0
        prev_node = 0
        current_node = tour[0]
        
        # The tour is a sequence of nodes. We start at depot (0).
        # We traverse the tour.
        # If adding next node exceeds capacity, we "return to depot" (add dist to depot)
        # and then "go from depot to next node" (add dist from depot).
        
        for i in range(len(tour)):
            node = tour[i]
            if node == 0:
                continue # Should not happen in NN tour of customers
                
            # Check if we can add this node to the current route
            if current_load + demands[node] > capacity:
                # Return to depot
                total_dist += distance_matrix[prev_node, 0]
                # Go from depot to this node
                total_dist += distance_matrix[0, node]
                current_load = demands[node]
                prev_node = node
            else:
                # Add node to current route
                total_dist += distance_matrix[prev_node, node]
                current_load += demands[node]
                prev_node = node
        
        # Return to depot at the end
        if prev_node != 0:
            total_dist += distance_matrix[prev_node, 0]
            
        return total_dist

    # Search for optimal lambda
    # Heuristic range for lambda:
    # If lambda is 0, it's pure distance.
    # If lambda is very large, high demand nodes are avoided (in the tour construction sense).
    # Max demand is 9. Min distance is ~0.1?
    # Let's try a range of lambdas.
    
    # We want to be efficient. 20-30 evaluations should be enough.
    # Lambda range: 0 to max_distance / min_demand?
    max_d = np.max(distance_matrix)
    min_d = np.min(demands[1:])
    # Scale lambda so that lam * min_d is comparable to max_d
    # lam_max = max_d / min_d
    # Let's sample log-spaced or linear.
    
    # Start with a small set of candidates
    # We know lambda=0 is a baseline.
    # Let's try values that penalize demand significantly.
    
    # Normalized demands: d / capacity?
    # Let's use absolute lambda.
    
    candidates = [0.0]
    # Add some geometric progression
    max_lam = max_d / max(min_d, 1e-9) * 10
    # Generate 20 points log spaced from 1e-3 * max_lam to max_lam
    if max_lam > 0:
        log_candidates = np.logspace(np.log10(1e-3 * max_lam), np.log10(max_lam), 20)
        candidates.extend(log_candidates.tolist())
    
    best_lam = 0.0
    best_score = np.inf
    
    for lam in candidates:
        score = eval_lambda(lam)
        if score < best_score:
            best_score = score
            best_lam = lam
            
    # Compute final heuristic using best_lam
    # H_ij = 1 / (D_ij + best_lam * d_j)
    
    H = distance_matrix + best_lam * demands
    # Handle division by zero or very small values
    H = np.maximum(H, 1e-9)
    H_inv = 1.0 / H
    
    # Ensure finite
    H_inv = np.where(np.isfinite(H_inv), H_inv, 1e-9)
    
    return H_inv
# EVOLVE-BLOCK-END
