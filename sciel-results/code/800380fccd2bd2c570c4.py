import numpy as np
from scipy.cluster.vq import kmeans2

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Heuristic matrix for ant colony optimization of capacitated vehicle routing.
    Uses K-Means clustering to identify geographic regions and boost intra-cluster edges.
    
    Args:
        distance_matrix: (n, n) matrix of Euclidean distances
        coordinates: (n, 2) array of coordinates in unit square
        demands: (n,) array of demands (node 0 is depot with demand 0)
        capacity: vehicle capacity (50)
    
    Returns:
        (n, n) edge-attractiveness matrix where larger values are more attractive.
        Non-positive entries become 1e-9.
    """
    d = np.array(distance_matrix, dtype=float)
    n = d.shape[0]
    
    if d.shape[0] != d.shape[1]:
        return np.ones((n, n))
    
    coords = np.array(coordinates, dtype=float)
    dem = np.array(demands, dtype=float)
    
    # Base heuristic: inverse distance
    base = np.zeros_like(d)
    mask = d > 1e-12
    base[mask] = 1.0 / d[mask]
    
    # Structural Change: K-Means Clustering Boost
    # Identify natural route regions
    if n > 1:
        # Use customers (1 to n-1) for clustering
        cust_coords = coords[1:]
        # Estimate K: roughly capacity / avg_demand. Avg demand ~ 5. Cap 50.
        # So ~ 10 routes? Or 5? Let's use 5 for 50 customers.
        K = 5
        # Ensure K is valid
        K = min(K, len(cust_coords))
        
        try:
            # kmeans2 is fast for small n
            # random_state is not available in older scipy? 
            # scipy.cluster.vq.kmeans2(data, k, minit='points', count=1, seed=None)
            # In newer scipy, seed is supported. Let's assume standard usage.
            # To be safe and deterministic if possible, we might need to handle seeds.
            # But the task says "seeded per instance" for the ACO, not necessarily the heuristic.
            # The heuristic is computed once. Randomness in K-Means might cause variance.
            # We can use 'points' for initialization to be somewhat stable, or just run it.
            # Let's use a fixed random seed for the heuristic calculation to ensure reproducibility
            # if the evaluator runs it multiple times? The task says "seeded per instance" for ACO.
            # It doesn't specify seeding the heuristic.
            # However, to avoid bad clustering, let's use 'points' initialization.
            
            centroids, labels = kmeans2(cust_coords, K, minit='points', count=1)
            
            # Map labels back to full node indices
            cluster_labels = np.zeros(n, dtype=int)
            cluster_labels[0] = -1 # Depot has no cluster
            cluster_labels[1:] = labels
            
            # Create same-cluster matrix
            same_cluster = (cluster_labels[:, np.newaxis] == cluster_labels[np.newaxis, :]).astype(float)
            
            # Do not boost edges involving the depot based on cluster identity
            # because the depot must visit all clusters.
            same_cluster[0, :] = 0.0
            same_cluster[:, 0] = 0.0
            
            # Boost factor: 1.0 for different clusters, 1.5 for same cluster
            # This makes intra-cluster edges 50% more attractive
            cluster_boost = 1.0 + 0.5 * same_cluster
            
        except Exception:
            # Fallback if clustering fails
            cluster_boost = np.ones((n, n))
    else:
        cluster_boost = np.ones((n, n))
        
    # Combine
    heuristic = base * cluster_boost
    
    # Ensure diagonal is very small
    np.fill_diagonal(heuristic, 1e-9)
    
    # Handle non-positive or non-finite values
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
