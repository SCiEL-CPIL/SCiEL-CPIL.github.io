"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.spatial as spatial
import time

# Global cache for MDS coordinates to avoid recomputation if possible, 
# though the distance matrix is fixed per instance, we can't easily key it.
# However, MDS on 50x50 is fast enough to do every time if optimized.
# To be safe against timeouts, we will compute MDS only if the number of unvisited nodes is large,
# otherwise fall back to NN.

def _mds_coordinates(dist_matrix):
    """
    Perform Classical Multidimensional Scaling to recover 2D coordinates 
    from a Euclidean distance matrix.
    """
    n = dist_matrix.shape[0]
    # Center the matrix
    # D^2 is the input. We need to convert to inner product matrix.
    # J = I - (1/n) * 11^T
    # B = -0.5 * J * D^2 * J
    # Then eigendecompose B.
    
    D_sq = dist_matrix ** 2
    
    # Centering
    # Use the formula: B = -0.5 * J * D_sq * J
    # To save memory and time, we can do:
    # 1. Mean of rows
    # 2. Subtract means
    # 3. Multiply by J again?
    
    # Efficient centering:
    # B = -0.5 * (D_sq - row_means - col_means + grand_mean)
    row_means = D_sq.mean(axis=1, keepdims=True)
    col_means = D_sq.mean(axis=0, keepdims=True)
    grand_mean = D_sq.mean()
    
    B = -0.5 * (D_sq - row_means - col_means + grand_mean)
    
    # Eigendecomposition
    # We only need the top 2 eigenvalues/vectors
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(B)
    except np.linalg.LinAlgError:
        # Fallback if numerical issues
        return None

    # Sort by eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Take top 2
    # Coordinates are V * sqrt(lambda)
    # We need to ensure real coordinates
    if eigenvalues[0] < 0 or eigenvalues[1] < 0:
        # If eigenvalues are negative (shouldn't be for Euclidean), take abs or return None
        # For Euclidean distance matrices, eigenvalues should be non-negative.
        # Small negative values are numerical noise.
        eigenvalues = np.maximum(eigenvalues, 0)
        
    coords = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])
    
    return coords

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a Convex Hull Peeling strategy based on MDS coordinates.
    """
    if len(unvisited_nodes) == 0:
        return -1 # Should not happen
        
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
        
    # If few nodes, NN is fast and good enough
    if len(unvisited_nodes) <= 5:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin for safety

    # Try to get coordinates
    coords = None
    try:
        coords = _mds_coordinates(distance_matrix)
        if coords is None:
            raise ValueError("MDS failed")
    except Exception:
        # Fallback to NN if MDS fails
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # Check time
    if time.time() > deadline:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    current_coord = coords[current_node]
    unvisited_coords = coords[unvisited_nodes]
    
    # Compute Convex Hull of unvisited nodes
    try:
        hull = spatial.ConvexHull(unvisited_coords)
        hull_vertices = unvisited_nodes[hull.vertices]
    except Exception:
        # If hull computation fails (e.g., collinear points), fallback to NN
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # Check time
    if time.time() > deadline:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # Determine if current_node is on the hull
    # We can check if current_coord is on the boundary of the hull defined by unvisited_coords
    # A simpler heuristic: 
    # 1. Find the closest point on the hull to current_node.
    # 2. If current_node is "inside" the hull, go to the closest hull point.
    # 3. If current_node is "outside" or on the hull, continue the hull sequence.
    
    # To distinguish inside/outside, we can check if current_node is in the convex hull of unvisited_nodes.
    # scipy.spatial.ConvexHull has a .contains_points method? No, it's not standard in older versions.
    # We can use scipy.spatial.Delaunay or check against the hull edges.
    # Simpler: 
    # If we assume the tour is being built, the current node was just visited.
    # The "Hull Peeling" logic: 
    # Always pick a node on the convex hull of the remaining nodes.
    # Which one? 
    # Option A: The one closest to current_node.
    # Option B: The one that maintains the angular sweep.
    
    # Let's use Option A (Nearest Hull Point) as it is robust and simple.
    # This is "Nearest Neighbor restricted to the Convex Hull".
    
    distances_to_hull = distance_matrix[current_node, hull_vertices]
    min_idx = np.argmin(distances_to_hull)
    next_node = hull_vertices[min_idx]
    
    # Optimization: If the current node is far from the hull, maybe going to the nearest *unvisited* node 
    # that is NOT on the hull is better? No, peeling says stay on the hull.
    
    # However, if we are stuck in the middle, we should go to the hull.
    # If we are on the hull, we should go to the next hull point.
    # The "Nearest Hull Point" strategy naturally handles both:
    # - If inside, it pulls you out to the nearest boundary.
    # - If on the boundary, it moves you to the adjacent boundary point (since it's the nearest).
    
    return int(next_node)

# EVOLVE-BLOCK-END
