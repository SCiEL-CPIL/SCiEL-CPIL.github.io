"""Improved angular sweep heuristic for capacitated vehicle routing."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Select the next node using an angular sweep heuristic around the depot.
    
    This heuristic minimizes the angular deviation between the vector (Depot -> Current)
    and (Depot -> Candidate), while penalizing the distance from the current node.
    This encourages the route to sweep around the depot in a consistent sector,
    reducing backtracking and route length.
    """
    depot = int(depot)
    
    # Filter unvisited nodes by capacity feasibility
    feasible = []
    for node in unvisited_nodes:
        node = int(node)
        if demands[node] <= rest_capacity:
            feasible.append(node)
    
    if not feasible:
        return 0  # Close route, return to depot
    
    # Pre-compute distance from depot to current node
    d_depot_current = distance_matrix[depot, current_node]
    
    best_node = feasible[0]
    best_score = float('inf')
    
    for node in feasible:
        node = int(node)
        d_depot_node = distance_matrix[depot, node]
        d_current_node = distance_matrix[current_node, node]
        
        # Compute angle at depot using law of cosines
        # a = d(depot, current), b = d(depot, node), c = d(current, node)
        a = d_depot_current
        b = d_depot_node
        c = d_current_node
        
        if a < 1e-12 or b < 1e-12:
            angle = 0.0
        else:
            cos_theta = (a*a + b*b - c*c) / (2.0 * a * b)
            cos_theta = max(-1.0, min(1.0, cos_theta))
            angle = np.arccos(cos_theta)
        
        # Improved Scoring:
        # 1. Angular penalty: Minimize the angle to stay in the same sector.
        # 2. Distance penalty: Minimize the actual travel distance from current node.
        # 3. Radius penalty: Prefer nodes with similar radius to current to maintain sweep consistency.
        
        # Normalize distance by average radius
        avg_radius = 0.5 * (a + b) + 1e-12
        dist_penalty = c / avg_radius
        
        # Radius difference penalty: penalize large jumps in radius
        radius_diff = abs(a - b) / (a + b + 1e-12)
        
        # Weighted combination with tuned weights
        # Higher weight on angle to enforce sector clustering
        # Moderate weight on distance to prefer nearby nodes
        # Small weight on radius difference to maintain smooth sweep
        score = 1.2 * angle + 0.3 * dist_penalty + 0.1 * radius_diff
        
        if score < best_score:
            best_score = score
            best_node = node
    
    return int(best_node)
# EVOLVE-BLOCK-END
