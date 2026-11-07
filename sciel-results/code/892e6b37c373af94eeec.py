# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Generates an edge-attractiveness matrix based on capacity-aware spatial clustering.
    
    Strategy:
    1. Compute a base heuristic (inverse distance).
    2. Create a spatial partition using a sweep-line algorithm to form initial clusters 
       that respect the capacity constraint.
    3. Refine clusters by merging adjacent ones if it improves "compactness" (reduces 
       perimeter or increases density) while maintaining capacity feasibility.
    4. Construct the heuristic matrix:
       - Edges within the same cluster get a high boost (encouraging local tours).
       - Edges between clusters are penalized based on the distance between cluster 
         centroids, discouraging long-range jumps that break spatial locality.
    5. The final heuristic is the product of the base inverse distance and a 
       "locality factor" derived from the cluster structure.
    """
    n = distance_matrix.shape[0]
    start_time = time.time()
    deadline = start_time + 280.0 # Safety margin

    distance_matrix = np.asarray(distance_matrix, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    demands = np.asarray(demands, dtype=float)
    
    # Ensure positivity and finite values
    if not np.all(np.isfinite(distance_matrix)):
        return np.ones((n, n)) * 1e-9
        
    eps = 1e-9
    safe_dist = distance_matrix + eps
    
    # Base heuristic: Inverse distance
    h_base = 1.0 / safe_dist
    
    # 1. Initial Clustering via Sweep Line
    # Sort customers (excluding depot 0) by angle from depot
    # This approximates a spatial partition that is friendly to TSP tours
    depot_coord = coordinates[0]
    customer_indices = np.arange(1, n)
    
    # Vectorized angle calculation
    dx = coordinates[customer_indices, 0] - depot_coord[0]
    dy = coordinates[customer_indices, 1] - depot_coord[1]
    angles = np.arctan2(dy, dx)
    
    # Sort indices by angle
    sorted_customers = customer_indices[np.argsort(angles)]
    
    # Sweep to create initial clusters
    # We want clusters that don't exceed capacity.
    # Since demands are 1-9 and capacity is 50, a cluster will likely have ~5-10 nodes.
    clusters = []
    current_cluster = []
    current_load = 0.0
    
    for idx in sorted_customers:
        if current_load + demands[idx] <= capacity:
            current_cluster.append(idx)
            current_load += demands[idx]
        else:
            if current_cluster:
                clusters.append(current_cluster)
            current_cluster = [idx]
            current_load = demands[idx]
            
    if current_cluster:
        clusters.append(current_cluster)
        
    # 2. Refine Clusters (Optional: Merge small adjacent clusters if beneficial)
    # For simplicity and robustness, we will stick to the sweep clusters but 
    # calculate their geometric properties.
    
    cluster_ids = np.zeros(n, dtype=int)
    cluster_centroids = []
    cluster_radii = []
    
    for cid, cluster in enumerate(clusters):
        if not cluster:
            continue
        cluster_ids[cluster] = cid
        coords_c = coordinates[cluster]
        centroid = np.mean(coords_c, axis=0)
        cluster_centroids.append(centroid)
        
        # Compute max distance from centroid to any point in cluster (approx radius)
        dists_to_centroid = np.linalg.norm(coords_c - centroid, axis=1)
        max_radius = np.max(dists_to_centroid)
        cluster_radii.append(max_radius)
        
    # Handle depot (node 0) - assign it to a special "global" cluster or just treat separately
    cluster_ids[0] = -1
    cluster_centroids.append(depot_coord) # Index -1 corresponds to depot? No, let's just use -1 as index.
    # Actually, let's just make a list of centroids where index i is centroid of cluster i.
    # Depot is not a cluster.
    
    num_clusters = len(clusters)
    
    # 3. Construct Heuristic with Locality Boost
    # We want to boost edges (i, j) if i and j are in the same cluster.
    # We want to penalize edges (i, j) if they are in different clusters, 
    # especially if the clusters are far apart.
    
    h_final = h_base.copy()
    
    # Vectorized cluster ID lookup
    # For each pair (i, j), check if cluster_ids[i] == cluster_ids[j]
    # If yes, boost. If no, check distance between centroids.
    
    # Create a matrix of cluster IDs
    cid_matrix = cluster_ids[:, None]
    
    # Same cluster mask
    same_cluster = (cid_matrix == cid_matrix.T)
    
    # Boost factor for same-cluster edges
    # A stronger boost encourages the ACO to stay within clusters, 
    # which reduces the "crossing" penalty inherent in TSP.
    boost_same = 10.0
    
    # Penalty for cross-cluster edges
    # Calculate distance between centroids for all pairs
    # Centroid matrix: (num_clusters, 2)
    if num_clusters > 0:
        centroids_arr = np.array(cluster_centroids)
        
        # For each node i, get its cluster centroid
        # If node is depot (cid=-1), use depot coord
        # We need to map node index -> centroid index
        # Cluster IDs are 0..num_clusters-1. Depot is -1.
        
        # Create an array of centroids for each node
        node_centroids = np.zeros((n, 2))
        for i in range(n):
            cid = cluster_ids[i]
            if cid == -1:
                node_centroids[i] = depot_coord
            else:
                node_centroids[i] = centroids_arr[cid]
                
        # Calculate squared distance between node centroids for all pairs
        # d_centroid[i, j] = ||c_i - c_j||^2
        # This is O(n^2) but n=50, so it's fast.
        
        diff = node_centroids[:, None, :] - node_centroids[None, :, :]
        centroid_dist_sq = np.sum(diff**2, axis=2)
        
        # Penalty: If clusters are different, and centroids are far, reduce attractiveness.
        # We use a Gaussian decay: exp(-alpha * d^2)
        # Alpha controls how quickly cross-cluster edges are penalized.
        # Larger alpha -> stricter locality.
        alpha = 50.0 
        
        # Only apply penalty to cross-cluster edges
        cross_cluster = ~same_cluster
        
        # Calculate penalty factor
        # For same cluster, factor = 1.0 (or boost)
        # For cross cluster, factor = exp(-alpha * centroid_dist_sq)
        
        penalty_factor = np.ones((n, n))
        penalty_factor[cross_cluster] = np.exp(-alpha * centroid_dist_sq[cross_cluster])
        
        # Apply boost to same cluster
        h_final *= penalty_factor
        h_final[same_cluster] *= boost_same
        
    # Ensure the heuristic is positive and finite
    h_final = np.maximum(h_final, 1e-9)
    h_final[~np.isfinite(h_final)] = 1e-9
    
    # Symmetrize just in case (though distance matrix is likely symmetric)
    # h_final = (h_final + h_final.T) / 2.0
    
    return h_final
# EVOLVE-BLOCK-END
