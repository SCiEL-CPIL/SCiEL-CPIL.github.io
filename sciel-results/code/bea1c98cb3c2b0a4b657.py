"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT = 115.0
_DEADLINE = _START_TIME + _TIMEOUT

# Cache for angular vectors relative to depot
_angle_cache = {}

def _get_node_angle(node, depot, distance_matrix):
    """Compute angle of node relative to depot using 2D reconstruction from distances."""
    if node in _angle_cache:
        return _angle_cache[node]
    
    # We need coordinates. Since we only have distance matrix, we'll use a trick:
    # For a 2D Euclidean space, we can reconstruct angles if we know coordinates.
    # But we don't have coordinates directly. 
    # However, the task says "distance_matrix" is provided. 
    # Let's check if we can assume Euclidean and use a simple proxy.
    # Actually, without coordinates, "angular sweep" is hard.
    # But wait, the distance matrix is 50x51 (depot + 50 nodes).
    # If the instances are Euclidean, we can try to recover coordinates? No, too slow/complex.
    
    # Alternative: Use the distance to depot and distance between nodes to estimate "direction".
    # A common heuristic in CVRP without coordinates is to use the "savings" or "insertion" logic.
    # The directive asks for "angular sweep". 
    # Perhaps we can approximate the angle by the change in distance to depot?
    # Or maybe the distance_matrix is actually derived from coordinates and we can assume 
    # that minimizing the angle between (current->next) and (current->depot) helps?
    
    # Let's try a different interpretation: 
    # "Minimize angular deviation from current route's prevailing direction"
    # Prevailing direction = direction from depot to current_node? Or from current_node to next?
    
    # Since we lack coordinates, let's use a proxy for "angle" based on distances.
    # In 2D Euclidean space, if we have points A, B, C:
    # Angle at B between A and C can be found via Law of Cosines.
    # Here, let's consider the triangle: Depot (D), Current (C), Candidate (N).
    # We want to keep moving in a "consistent" direction away from the depot or along a sweep.
    
    # Let's compute the angle at Current (C) between Depot (D) and Candidate (N).
    # d_DC = dist(D, C)
    # d_CN = dist(C, N)
    # d_DN = dist(D, N)
    # cos(theta) = (d_DC^2 + d_CN^2 - d_DN^2) / (2 * d_DC * d_CN)
    
    # If we want to "sweep" around the depot, we might want to keep the angle at the DEPOT consistent?
    # Angle at Depot (D) between Current (C) and Candidate (N).
    # d_DC, d_DN, d_CN.
    # cos(phi) = (d_DC^2 + d_DN^2 - d_CN^2) / (2 * d_DC * d_DN)
    
    # A "sweep" usually implies visiting nodes in angular order around the depot.
    # So we want the angle at the Depot between Current and Next to be small?
    # Or we want to continue the sweep, so the angle between the vector (Depot->Current) and (Depot->Next) should be small?
    
    # Let's compute the angle at the Depot.
    # This requires d_DC, d_DN, d_CN. All are in the distance matrix.
    
    d_dc = distance_matrix[depot, current_node]
    if d_dc < 1e-9:
        # Current is depot, angle is undefined/zero
        return 0.0
    
    # We will compute this on the fly for feasible nodes to avoid caching complexity if nodes change,
    # but since the set of nodes is fixed, we can cache the depot-angle for each node relative to the *previous* node?
    # No, the angle depends on the current node. So we can't cache a static angle per node.
    # We must compute the angle for each candidate relative to the current node.
    
    # This calculation is O(1) per candidate, which is fine for N=50.
    return None # Placeholder, logic moved to select_next_node

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Deadline guard
    if time.time() > _DEADLINE:
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return 0
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])

    # Filter feasible nodes
    feasible_mask = np.array([demands[n] <= rest_capacity for n in unvisited_nodes])
    if not np.any(feasible_mask):
        return 0
    
    feasible_nodes = unvisited_nodes[feasible_mask]
    
    # Distances
    dists_current = distance_matrix[current_node, feasible_nodes]
    dists_depot = distance_matrix[feasible_nodes, depot]
    dists_depot_current = distance_matrix[depot, current_node]
    
    # Heuristic Selection
    if current_node == depot:
        # Start with nearest neighbor to establish a direction
        best_idx = np.argmin(dists_current)
    else:
        # Angular Sweep Heuristic
        # We want to minimize the angle at the Depot between Current and Candidate.
        # This encourages visiting nodes in a circular sweep around the depot.
        
        # Law of Cosines for angle at Depot (phi):
        # cos(phi) = (d_DC^2 + d_DN^2 - d_CN^2) / (2 * d_DC * d_DN)
        # We want to MINIMIZE phi, so we want to MAXIMIZE cos(phi).
        # However, we also need to consider the distance to current node (efficiency).
        
        # Let's compute cos(phi) for each feasible node
        # d_DC is scalar
        # d_DN is dists_depot
        # d_CN is dists_current
        
        # Avoid division by zero
        denom = 2.0 * dists_depot_current * dists_depot
        denom[denom < 1e-9] = 1e-9
        
        # Numerator: d_DC^2 + d_DN^2 - d_CN^2
        numer = dists_depot_current**2 + dists_depot**2 - dists_current**2
        
        cos_phi = numer / denom
        
        # Clip to [-1, 1] for safety
        cos_phi = np.clip(cos_phi, -1.0, 1.0)
        
        # We want to maximize cos_phi (minimize angle).
        # But we also want to be close to current node.
        # Let's blend: Score = -cos_phi * beta + dists_current * (1-beta)
        # Or: Score = dists_current - gamma * cos_phi
        
        # Let's try: Minimize (dists_current - gamma * cos_phi)
        # High cos_phi (small angle) is good (negative term reduces score).
        # Small dists_current is good.
        
        gamma = 1.5 # Weight for angular consistency
        
        scores = dists_current - gamma * cos_phi
        best_idx = np.argmin(scores)
        
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
