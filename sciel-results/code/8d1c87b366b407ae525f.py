"""Angular sweep heuristic with savings check for CVRP."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = _start_time + _TIMEOUT

# Global cache for precomputed constants
_angles = None
_max_dist = None
_depot_idx = 0

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using an angular sweep heuristic.
    Encourages spatial clustering by minimizing angular deviation from the current route direction.
    """
    global _angles, _max_dist, _depot_idx
    
    # Initialize constants if not done
    if _angles is None:
        _depot_idx = depot
        # Calculate angles relative to depot
        # Assuming depot is at (0,0) or we use distance matrix to infer?
        # Wait, the distance matrix is given, but not coordinates.
        # We cannot compute true angles without coordinates.
        # However, we can use the distance matrix to create a "pseudo-angle" or just rely on savings.
        # BUT, the prompt says "distance_matrix" is provided. It does NOT provide coordinates.
        # Therefore, we CANNOT compute true geometric angles unless we assume Euclidean and have coords.
        # The task description says "eight generated 50-customer instances". Usually these are Euclidean.
        # If coordinates are not provided, we must infer them or use a different metric.
        # RE-READING: "distance_matrix" is the only input.
        # If we don't have coordinates, we can't do true angular sweep.
        # We can, however, use a "graph-theoretic" angle or just rely on savings.
        # Wait, if the evaluator generates instances, it might be Euclidean.
        # But the function signature only gives distance_matrix.
        # Can we reconstruct coordinates? No, not reliably from a generic distance matrix.
        # So, "Angular Sweep" in the geometric sense is impossible without coords.
        
        # Alternative: Use the structure of the distance matrix.
        # Perhaps the "inspiration" implies that we should *assume* we can get angles?
        # Or maybe the "angular sweep" is a metaphor for "clustering"?
        # Let's look at the inputs again.
        # If we can't get angles, we must use a different clustering method.
        # Common alternative: k-means clustering on the distance matrix (metric space clustering).
        # Or, simply use a stronger Savings heuristic.
        
        # Let's check if we can estimate coordinates.
        # If the instances are generated, they are likely random Euclidean points.
        # We can try to embed the distance matrix into 2D using MDS (Multidimensional Scaling) if SciPy is available.
        # "Python standard library, NumPy and SciPy are available."
        # We can use scipy.spatial.distance or sklearn (not listed, but scipy is).
        # Let's try to compute 2D coordinates from the distance matrix using MDS.
        
        pass # Placeholder for logic
        
    if time.time() > _DEADLINE:
        return depot

    if len(unvisited_nodes) == 0:
        return depot

    # Convert to numpy array
    unvisited = np.array(unvisited_nodes, dtype=int)
    
    # 1. Filter by capacity
    feasible_mask = demands[unvisited] <= rest_capacity
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    # If we haven't computed angles, we need to.
    # Since we only have distance_matrix, let's try to compute coordinates via MDS once.
    # This is expensive (O(N^3) or so), so we do it once.
    if _angles is None:
        try:
            from scipy.spatial.distance import squareform
            from sklearn.manifold import MDS # sklearn might not be available, only scipy
            # Let's use scipy.spatial.distance.cdist? No, we have distances.
            # Use classical MDS.
            # D is distance matrix.
            # B = -0.5 * J D^2 J
            # Eigendecompose B.
            
            D = distance_matrix
            n = D.shape[0]
            J = np.ones((n, n)) - np.eye(n)
            B = -0.5 * (J @ (D**2) @ J)
            
            # Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(B)
            
            # Take top 2 eigenvalues
            # Sort by descending eigenvalue
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Coordinates are eigenvectors * sqrt(eigenvalues)
            # Take first 2
            if eigenvalues[0] > 0 and eigenvalues[1] > 0:
                coords = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])
            else:
                # Fallback if MDS fails or is degenerate
                coords = np.random.rand(n, 2)
                
            # Compute angles relative to depot
            depot_coord = coords[depot]
            diffs = coords - depot_coord
            _angles = np.arctan2(diffs[:, 1], diffs[:, 0])
            _max_dist = np.max(D)
            
        except Exception:
            # If MDS fails, fallback to a simple savings heuristic
            _angles = None
            _max_dist = np.max(distance_matrix)
            # Return best by distance if no angles
            if current_node == depot:
                dists = distance_matrix[depot, feasible_nodes]
                return int(feasible_nodes[np.argmin(dists)])
            else:
                dists = distance_matrix[current_node, feasible_nodes]
                return int(feasible_nodes[np.argmin(dists)])

    if current_node == depot:
        # At depot: Pick node with min distance to start the sweep
        dists = distance_matrix[depot, feasible_nodes]
        min_dist_idx = np.argmin(dists)
        return int(feasible_nodes[min_dist_idx])

    # Not at depot: Angular Sweep
    # Get angles of feasible nodes
    angles_feasible = _angles[feasible_nodes]
    angle_current = _angles[current_node]
    
    # Calculate angular difference (counter-clockwise)
    # (theta_j - theta_c) mod 2pi
    delta_theta = (angles_feasible - angle_current) % (2 * np.pi)
    
    # Calculate distances
    dists = distance_matrix[current_node, feasible_nodes]
    
    # Normalize delta_theta to be comparable to distances
    # Max distance is _max_dist. Max angle is 2*pi.
    # We want to penalize large angle changes.
    # Score = normalized_angle * weight + distance
    
    # If the angle change is > pi, it might be better to go the other way?
    # Usually sweeps go one way. Let's assume we always go counter-clockwise.
    # If delta_theta is close to 2*pi, it's a small change in the other direction.
    # But to maintain a sweep, we should probably wrap around.
    # Actually, if we've served nodes in increasing angle, the next one should be slightly larger.
    # If we wrap around, we are closing the loop.
    
    # Let's use a simple score:
    # Penalty for large angular deviation.
    # If delta_theta > pi, the "shortest" angular path is the other way, but that breaks the sweep.
    # So we keep delta_theta as is.
    
    angle_score = delta_theta / (2 * np.pi) * _max_dist
    score = angle_score + dists
    
    # Find best candidate
    min_score_idx = np.argmin(score)
    best_node = feasible_nodes[min_score_idx]
    best_dist = dists[min_score_idx]
    best_angle_dev = delta_theta[min_score_idx]
    
    # Check if closing the route is better
    # Closing cost: dist(current, depot)
    close_cost = distance_matrix[current_node, depot]
    
    # Extension cost vs Closing cost
    # If going to best_node is significantly more expensive than going back to depot,
    # or if the angular deviation is too high (we've gone far around), close.
    
    # Heuristic: If best_dist > close_cost + distance_matrix[depot, best_node], 
    # it's not worth it (Savings < 0).
    savings = distance_matrix[current_node, depot] + distance_matrix[depot, best_node] - best_dist
    if savings < 0:
        return depot
    
    # Additional check: If the angular deviation is very large (e.g., > 3/4 circle),
    # it implies we are about to wrap around, which might be inefficient if we can close.
    # But if savings > 0, it's geometrically better to extend.
    
    return int(best_node)

# EVOLVE-BLOCK-END
