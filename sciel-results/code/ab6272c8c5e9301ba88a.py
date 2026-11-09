import numpy as np
from scipy.cluster.vq import kmeans2

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes a cluster-centric heuristic matrix for ACO-based CVRP.
    
    The heuristic encourages ants to visit customers within the same spatial cluster
    consecutively, reducing long inter-cluster transitions.
    """
    d = np.array(distance_matrix, dtype=float)
    coords = np.array(coordinates, dtype=float)
    demands = np.array(demands, dtype=float)
    
    n = d.shape[0]
    
    # 1. Determine number of clusters based on total demand and capacity
    total_demand = np.sum(demands[1:])  # Exclude depot demand
    if total_demand <= 0:
        K = 1
    else:
        K = int(np.ceil(total_demand / capacity))
        # Cap K to avoid too many tiny clusters which might fragment routes
        # For 50 customers, K should be reasonable, e.g., 3-10
        K = max(1, min(K, 10)) 
        
    # 2. Perform K-Means clustering on customer coordinates (excluding depot)
    # coords[1:] are the customer coordinates
    customer_coords = coords[1:]
    
    # K-Means requires K <= number of points
    if K > len(customer_coords):
        K = len(customer_coords)
        
    try:
        # kmeans2 returns centroids and assignments
        # seed=0 for reproducibility if needed, but heuristic is static per instance
        centroids, cluster_ids = kmeans2(customer_coords, K, minit='points', seed=42)
    except Exception:
        # Fallback if kmeans fails (e.g., K too high, numerical issues)
        cluster_ids = np.zeros(len(customer_coords), dtype=int)
        
    # Map cluster IDs back to full node indices (0 is depot)
    # Depot (0) is assigned to a special cluster ID (-1)
    full_cluster_ids = np.full(n, -1, dtype=int)
    full_cluster_ids[1:] = cluster_ids
    
    # 3. Construct Heuristic Matrix
    # Base heuristic: Inverse distance
    # We use a power > 1 to sharpen local preference
    eps = 1e-9
    d_safe = d + eps
    
    # Initialize with inverse distance (standard ACO heuristic)
    # Using power 2 for a stronger local bias than power 1
    heuristic = 1.0 / (d_safe ** 2)
    
    # Apply Cluster Boost
    # Edges within the same cluster get a significant boost
    # This encourages "sweeping" a cluster before leaving it
    BOOST_FACTOR = 10.0 
    
    # Create a boolean mask for intra-cluster edges (excluding depot)
    # i and j are in the same cluster, and neither is the depot
    intra_cluster_mask = (full_cluster_ids[:, np.newaxis] == full_cluster_ids[np.newaxis, :])
    # Exclude edges involving the depot (index 0) from the boost logic
    # We want depot edges to be driven purely by distance to ensure
    # the ant can start/end routes efficiently
    depot_mask = (np.arange(n) == 0)
    intra_cluster_mask = intra_cluster_mask & ~depot_mask[:, np.newaxis] & ~depot_mask[np.newaxis, :]
    
    # Apply boost
    heuristic[intra_cluster_mask] *= BOOST_FACTOR
    
    # 4. Post-processing
    # Set diagonal to 0 (or low value)
    np.fill_diagonal(heuristic, 0.0)
    
    # Ensure non-positive values become 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
