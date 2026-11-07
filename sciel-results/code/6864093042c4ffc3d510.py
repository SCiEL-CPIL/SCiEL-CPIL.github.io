import numpy as np
import time
import random

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Returns an edge-attractiveness matrix.
    Strategy: Global Cluster-Partition Heuristic.
    
    1. Partition customers into capacitated clusters using a fast greedy spatial sweep.
    2. The heuristic H[i, j] is defined as:
       - If i and j are in the same cluster: High base weight (encourages finishing the zone)
       - If i and j are in different clusters: Weight decays with the distance between cluster centroids.
       
    This structural change moves from local edge-based biases to a global zone-completion bias,
    which should reduce the "zig-zag" across the entire domain and improve capacity utilization
    by keeping loads localized within spatial regions.
    """
    start_time = time.time()
    deadline = start_time + 315.0  # 15s safety margin before 330s kill
    
    D = np.asarray(distance_matrix, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)
    n = D.shape[0]
    
    # 1. Precompute clusters
    # We use a simple greedy sweep: sort customers by angle or x-coordinate, 
    # or just use a nearest-neighbor chain to form clusters.
    # Given n=50, a simple "fill current cluster with nearest neighbors until full" is fast.
    
    # To make it robust to geometry, let's use a k-means-like assignment but with capacity constraints.
    # However, k-means doesn't respect capacity. 
    # Let's use a "Sweep" heuristic:
    # 1. Start with depot.
    # 2. Pick the unvisited customer closest to the current "center of mass" of the active cluster.
    # 3. Add it if it fits.
    # 4. If no one fits or all visited, close cluster, start new one from nearest unvisited to depot? 
    #    Actually, standard CVRP heuristics often use "Savings" or "Clarke-Wright".
    #    Let's try a simple spatial partition: Divide the unit square into a grid, or just use K-Medoids with capacity repair.
    
    # Simpler and faster for n=50: 
    # Use a "Greedy Nearest Neighbor Chain" to form routes/clusters.
    # This creates a set of "zones" (clusters) that are spatially coherent.
    
    unvisited = set(range(1, n))
    clusters = []
    cluster_ids = np.zeros(n, dtype=int)
    cluster_id = 0
    
    while unvisited:
        # Start new cluster from the customer closest to the depot among unvisited
        # Or just pick the first unvisited? 
        # Better: Pick the unvisited customer closest to the previous cluster's centroid?
        # For the first cluster, closest to depot.
        if not clusters:
            # Closest to depot (0)
            dists_to_depot = D[0, list(unvisited)]
            min_idx = np.argmin(dists_to_depot)
            start_node = list(unvisited)[min_idx]
        else:
            # Closest to the last cluster's centroid
            last_cluster = clusters[-1]
            centroid = np.mean(coords[last_cluster], axis=0)
            dists_to_centroid = np.linalg.norm(coords[list(unvisited)] - centroid, axis=1)
            min_idx = np.argmin(dists_to_centroid)
            start_node = list(unvisited)[min_idx]
            
        current_cluster = [start_node]
        unvisited.remove(start_node)
        current_load = demands[start_node]
        
        # Try to add more nodes to this cluster
        # We want to fill it up to capacity to reduce the number of clusters (trips)
        while unvisited:
            # Find the unvisited node with smallest demand that fits? 
            # Or the one that minimizes distance increase?
            # Let's pick the one that minimizes distance to the current cluster's "center" 
            # while fitting in capacity.
            feasible = [j for j in unvisited if demands[j] <= (capacity - current_load)]
            if not feasible:
                break
            
            # Calculate centroid of current cluster
            centroid = np.mean(coords[current_cluster], axis=0)
            dists = np.linalg.norm(coords[feasible] - centroid, axis=1)
            min_idx = np.argmin(dists)
            next_node = feasible[min_idx]
            
            current_cluster.append(next_node)
            unvisited.remove(next_node)
            current_load += demands[next_node]
            
            # Stop if we've filled the cluster or if no more fit
            if current_load >= capacity * 0.95: # Heuristic: fill ~95%
                break
            # Also stop if the next best is too far? 
            # For now, just fill by proximity.
            
        clusters.append(current_cluster)
        for node in current_cluster:
            cluster_ids[node] = cluster_id
        cluster_id += 1
        
    # Depot (0) is assigned to a special "Depot Cluster" or we handle it separately.
    # Let's assign depot to cluster -1 or 0? 
    # In the heuristic, edges from depot should probably encourage going to the "first" cluster?
    # Or just use distance for depot edges?
    # Let's assign depot to cluster 0 for simplicity, but its "centroids" logic might be skewed.
    # Better: Treat depot as neutral.
    
    # 2. Compute Cluster Centroids
    num_clusters = len(clusters)
    centroids = np.zeros((num_clusters, 2))
    for cid, cluster in enumerate(clusters):
        centroids[cid] = np.mean(coords[cluster], axis=0)
    
    # 3. Construct Heuristic Matrix
    # H[i, j] = base * cluster_bonus
    # base = 1 / (d_ij + epsilon)
    # cluster_bonus:
    #   if cluster_ids[i] == cluster_ids[j]: 1.0 (High)
    #   else: exp(-lambda * dist_centroid[i, j]) (Low, decays with cluster separation)
    #   if i == 0 or j == 0: 1.0 (Neutral, rely on distance)
    
    # We need to search for lambda (decay rate) and maybe a boost factor for intra-cluster.
    
    def simulate_aco(heuristic_matrix):
        """
        Runs the EXACT ACO protocol.
        """
        pheromone = np.ones_like(heuristic_matrix)
        best_total_cost = float('inf')
        
        num_ants = 20
        num_iterations = 30
        
        for _ in range(num_iterations):
            iteration_best_cost = float('inf')
            
            for ant in range(num_ants):
                current = 0
                current_load = 0.0
                total_dist = 0.0
                unvisited = set(range(1, n))
                
                while unvisited:
                    feasible = [j for j in unvisited if demands[j] <= (capacity - current_load)]
                    
                    if not feasible:
                        total_dist += D[current, 0]
                        break
                    
                    feasible_arr = np.array(feasible, dtype=int)
                    weights = pheromone[current, feasible_arr] * heuristic_matrix[current, feasible_arr]
                    
                    total_w = np.sum(weights)
                    if total_w < 1e-10:
                        next_node = random.choice(feasible)
                    else:
                        probs = weights / total_w
                        next_node = random.choices(feasible, weights=probs, k=1)[0]
                    
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
            
            pheromone *= 0.9
            if iteration_best_cost > 0:
                pheromone += 1.0 / iteration_best_cost
        
        return best_total_cost

    best_cost = float('inf')
    best_heuristic = None
    
    # Precompute distance between centroids for fast heuristic construction
    # dist_centroids[cid1, cid2]
    centroid_dists = np.zeros((num_clusters, num_clusters))
    for i in range(num_clusters):
        for j in range(num_clusters):
            centroid_dists[i, j] = np.linalg.norm(centroids[i] - centroids[j])
    
    # Search parameters:
    # lambda: decay rate for inter-cluster distance
    # boost: multiplier for intra-cluster edges
    
    candidates = [
        (1.0, 1.0),   # Mild decay, no boost
        (2.0, 1.5),   # Moderate decay, some boost
        (3.0, 2.0),   # Strong decay, high boost
        (1.5, 3.0),   # Mild decay, very high boost
        (2.5, 1.0),   # Strong decay, no boost
        (4.0, 2.5),   # Very strong decay, high boost
    ]
    
    random.shuffle(candidates)
    
    for lam, boost in candidates:
        if time.time() > deadline - 10.0:
            break
            
        H = np.zeros((n, n), dtype=np.float64)
        
        # Base distance inverse
        with np.errstate(divide='ignore', invalid='ignore'):
            base = np.where(D > 1e-10, 1.0 / D, 0.0)
        
        H = base.copy()
        
        for i in range(1, n):
            for j in range(1, n):
                if i == j:
                    continue
                ci = cluster_ids[i]
                cj = cluster_ids[j]
                
                if ci == cj:
                    H[i, j] *= boost
                else:
                    # Decay based on centroid distance
                    # Normalize centroid distance? Max dist in unit square is ~1.41
                    # Let's use exp(-lam * dist)
                    decay = np.exp(-lam * centroid_dists[ci, cj])
                    H[i, j] *= decay
        
        # Handle Depot (0)
        # For depot edges, we might want to bias towards the "closest" cluster?
        # Or just keep base distance?
        # Let's keep base distance for depot to avoid biasing the start too much,
        # but maybe boost the first cluster? 
        # For simplicity, keep depot edges as base distance (1/d).
        # H[0, :] and H[:, 0] are already base.
        
        # Sanitize
        H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
        
        cost = simulate_aco(H)
        
        if cost < best_cost:
            best_cost = cost
            best_heuristic = H

    if best_heuristic is None:
        with np.errstate(divide='ignore', invalid='ignore'):
            best_heuristic = np.where(D > 0, D ** (-1.0), 1e-9)
        best_heuristic = np.where(np.isfinite(best_heuristic) & (best_heuristic > 0), best_heuristic, 1e-9)

    return best_heuristic
