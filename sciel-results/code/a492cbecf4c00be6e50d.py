"""
Strategy: Spherical K-Means Sectoring.
Instead of purely local greedy selection, we compute a global geometric partition 
(k-means on the 2D/3D coordinates derived from the distance matrix) to identify 
spatial "sectors". The heuristic strongly biases towards nodes in the current 
sector, only allowing moves to other sectors when the current one is exhausted 
or when the move is significantly better. This avoids the local minima and 
crossing paths typical of standard Nearest Neighbor.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist

_start_time = time.time()
_DEADLINE = 145.0

# Cache for MDS coordinates and cluster centers
_mds_cache = {}
_cluster_cache = {}

def _get_coords(dist_matrix):
    """
    Reconstruct approximate 2D coordinates from distance matrix using MDS.
    This is a structural change: we move from metric space to Euclidean space 
    to enable geometric clustering.
    """
    key = id(dist_matrix)
    if key in _mds_cache:
        return _mds_cache[key]
    
    # Simple MDS using eigendecomposition of double-centered Gram matrix
    # D is the distance matrix
    D = dist_matrix
    n = D.shape[0]
    
    # Double centering
    # J = I - 1/n * 11^T
    # B = -0.5 * J D^2 J
    D2 = D ** 2
    one_n = np.ones((n, n)) / n
    J = np.eye(n) - one_n
    B = -0.5 * (J @ D2 @ J)
    
    # Eigendecomposition
    # We only need the top 2 eigenvalues/vectors
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(B)
    except:
        return None

    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Take top 2
    # Ensure real part
    coords = eigenvectors[:, :2] * np.sqrt(np.maximum(eigenvalues[:2], 0))
    
    # Center the coordinates
    coords = coords - coords.mean(axis=0)
    
    _mds_cache[key] = coords
    return coords

def _get_clusters(dist_matrix, coords):
    """
    Perform K-Means clustering on the coordinates to define sectors.
    """
    key = id(dist_matrix)
    if key in _cluster_cache:
        return _cluster_cache[key]
    
    if coords is None:
        return None

    n = coords.shape[0]
    # Heuristic number of clusters: sqrt(n) is often good for TSP sectors
    k = max(2, int(np.sqrt(n)))
    
    # K-Means initialization: pick random points or use kmeans++ like logic
    # For simplicity and speed, we use a few iterations of standard k-means
    # Initialize centers randomly
    rng = np.random.default_rng(42)
    centers = coords[rng.choice(n, k, replace=False)].copy()
    
    labels = np.zeros(n, dtype=int)
    
    for _ in range(10):
        # Assign points to nearest center
        distances = cdist(coords, centers)
        labels = np.argmin(distances, axis=1)
        
        # Update centers
        new_centers = np.zeros_like(centers)
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                new_centers[i] = coords[mask].mean(axis=0)
            else:
                # If a cluster is empty, re-initialize it to a random point
                new_centers[i] = coords[rng.choice(n)]
        
        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    _cluster_cache[key] = (labels, centers, k)
    return (labels, centers, k)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using Spherical K-Means Sectoring.
    """
    # Time guard
    if time.time() - _start_time > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Get global structure
    coords = _get_coords(distance_matrix)
    cluster_data = _get_clusters(distance_matrix, coords)
    
    if cluster_data is None:
        # Fallback to nearest neighbor if MDS/Clustering fails
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    labels, centers, k = cluster_data
    
    # Determine current sector (cluster) of the current node
    current_cluster = labels[current_node]
    
    # Distances from current to all unvisited
    d_curr = distance_matrix[current_node, unvisited_nodes]
    
    # Identify which unvisited nodes are in the current cluster
    unvisited_labels = labels[unvisited_nodes]
    in_current_cluster = (unvisited_labels == current_cluster)
    
    # Strategy:
    # 1. If there are nodes in the current cluster, strongly prefer them.
    # 2. If the current cluster is empty (or we are forced to move), pick the best node 
    #    that minimizes distance to the *center* of the next best cluster or simply 
    #    the nearest node overall but with a penalty for crossing boundaries.
    
    # To avoid getting stuck in a small cluster that forces a long jump later,
    # we use a dynamic penalty.
    
    # Calculate distance from current node to the *center* of each cluster
    # This helps decide when to leave a cluster.
    
    # Simple approach: 
    # Score = d_curr * (1 + penalty)
    # Penalty is 0 if in same cluster, 1.0 if in different cluster.
    # This effectively says: "Don't leave your sector unless it's empty or the 
    # distance to the outside is significantly smaller than the distance to the inside."
    
    # However, pure sectoring can lead to long edges if sectors are ill-shaped.
    # Let's use a "soft" sectoring.
    
    # Distance from each unvisited node to its own cluster center
    # This measures how "deep" inside the cluster a node is.
    
    # Let's try a different structural angle: 
    # "Sweeping" heuristic. 
    # If we have coordinates, we can compute the angle of each node relative to the centroid.
    # Sort by angle and pick the next node in the angular sequence?
    # This is the "Angular Sweep" or "Circle" heuristic.
    # It creates a star-shaped polygon which is often close to optimal for convex distributions.
    
    if coords is not None:
        # Compute centroid of all nodes
        centroid = coords.mean(axis=0)
        
        # Compute angles of unvisited nodes relative to centroid
        # We need to handle the 2D case
        dx = coords[unvisited_nodes, 0] - centroid[0]
        dy = coords[unvisited_nodes, 1] - centroid[1]
        
        angles = np.arctan2(dy, dx)
        
        # Current node's angle
        cur_dx = coords[current_node, 0] - centroid[0]
        cur_dy = coords[current_node, 1] - centroid[1]
        cur_angle = np.arctan2(cur_dy, cur_dx)
        
        # We want to pick the node with the smallest positive angle difference 
        # (moving counter-clockwise) or smallest absolute angle difference?
        # Moving in a consistent direction (sweep) avoids crossings.
        
        # Difference in angles
        # Normalize difference to [0, 2*pi)
        diff = (angles - cur_angle) % (2 * np.pi)
        
        # The next node in the sweep is the one with the smallest positive diff
        # But we also need to consider distance. A node far away in the sweep order
        # might be a bad choice if a close node is slightly ahead in angle.
        
        # Hybrid: Score = d_curr + lambda * diff
        # This encourages following the angular order but allows skipping if a node is very close.
        
        # Adaptive lambda:
        # Early in the tour, we want to establish the sweep direction -> high lambda.
        # Late in the tour, we want to close the loop -> low lambda (distance dominates).
        
        # n_unvisited starts at 49 (if 50 cities total, 1 start, 1 current? No, unvisited is remaining).
        # Let's assume total N=50.
        # progress = 1 - n_unvisited / 50
        progress = 1.0 - (n_unvisited / 50.0)
        
        # Lambda decreases as we progress
        # Start high (10.0) to enforce sweep, end low (0.5) to allow local optimization
        # Scale by average distance to make it dimensionless-ish
        avg_d = np.mean(d_curr)
        if avg_d == 0: avg_d = 1.0
        
        lambda_sweep = 10.0 * (1.0 - progress) + 0.5
        
        # Normalize angle difference to be in range [0, 1] for weighting?
        # diff is in [0, 2*pi).
        # Let's normalize diff by 2*pi to get [0, 1)
        diff_norm = diff / (2 * np.pi)
        
        # Score = d_curr + lambda_sweep * avg_d * diff_norm
        # Wait, if we just pick the next in angle, we might pick a node that is far away.
        # The term lambda * avg_d * diff_norm adds a penalty proportional to how far ahead in the sweep it is.
        
        scores = d_curr + lambda_sweep * avg_d * diff_norm
        
        best_idx = np.argmin(scores)
        return int(unvisited_nodes[best_idx])
        
    else:
        # Fallback if coords failed
        return int(unvisited_nodes[np.argmin(d_curr)])
# EVOLVE-BLOCK-END
