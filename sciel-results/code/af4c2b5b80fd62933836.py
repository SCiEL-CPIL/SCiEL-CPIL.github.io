"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes a heuristic matrix for CVRP ACO.
    Uses a Cluster-First Route-Second strategy:
    1. Partition customers into capacity-feasible clusters using a sweep algorithm.
    2. Solve the exact TSP for each cluster (including depot) using Held-Karp.
    3. Boost the heuristic values for edges that are part of these optimal cluster tours.
    """
    n = len(demands)
    depot = 0
    customers = list(range(1, n))
    
    # Base heuristic: inverse distance
    # Avoid division by zero
    dm = np.array(distance_matrix, dtype=float)
    base_heur = 1.0 / (dm + 1e-9)
    
    # 1. Clustering using Sweep Algorithm
    # Calculate angles of customers relative to depot
    depot_coords = coordinates[depot]
    angles = []
    for i in customers:
        dx = coordinates[i][0] - depot_coords[0]
        dy = coordinates[i][1] - depot_coords[1]
        angle = np.arctan2(dy, dx)
        angles.append((angle, i))
    
    # Sort by angle
    angles.sort(key=lambda x: x[0])
    sorted_customers = [idx for _, idx in angles]
    
    clusters = []
    current_cluster = []
    current_load = 0
    
    for i in sorted_customers:
        d = demands[i]
        if current_load + d <= capacity:
            current_cluster.append(i)
            current_load += d
        else:
            if current_cluster:
                clusters.append(current_cluster)
            current_cluster = [i]
            current_load = d
            
    if current_cluster:
        clusters.append(current_cluster)
        
    # 2. Solve TSP for each cluster and collect edges
    # We will boost the heuristic for edges in these tours
    boost_factor = 100.0
    
    for cluster in clusters:
        if not cluster:
            continue
            
        # Nodes in this sub-problem: Depot + cluster nodes
        sub_nodes = [depot] + cluster
        m = len(sub_nodes)
        
        # Extract sub-distance matrix
        # sub_dist[i][j] is distance between sub_nodes[i] and sub_nodes[j]
        sub_dist = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                sub_dist[i][j] = dm[sub_nodes[i]][sub_nodes[j]]
                
        # Held-Karp DP for TSP
        # dp[mask][last] = min cost to visit set 'mask' ending at 'last'
        # mask is a bitmask of nodes visited in sub_nodes
        # We start at node 0 (depot) in sub_nodes
        # We must return to node 0
        
        # Initialize DP
        # m is small (<= ~10-15 usually), so 2^m is manageable
        # If m > 20, Held-Karp is too slow, but with C=50 and avg demand 5, m is small.
        # Fallback to Nearest Neighbor if m is too large
        if m > 18:
            # Fallback: Nearest Neighbor
            visited = [0]
            current = 0
            unvisited = set(range(1, m))
            while unvisited:
                best_next = -1
                best_dist = float('inf')
                for j in unvisited:
                    if sub_dist[current][j] < best_dist:
                        best_dist = sub_dist[current][j]
                        best_next = j
                if best_next == -1:
                    break
                # Record edge current -> best_next
                orig_i = sub_nodes[current]
                orig_j = sub_nodes[best_next]
                base_heur[orig_i][orig_j] = max(base_heur[orig_i][orig_j], boost_factor / sub_dist[orig_i][orig_j])
                base_heur[orig_j][orig_i] = max(base_heur[orig_j][orig_i], boost_factor / sub_dist[orig_j][orig_i])
                
                visited.append(best_next)
                unvisited.remove(best_next)
                current = best_next
            # Return to depot
            orig_i = sub_nodes[current]
            orig_j = sub_nodes[0]
            base_heur[orig_i][orig_j] = max(base_heur[orig_i][orig_j], boost_factor / sub_dist[orig_i][orig_j])
            base_heur[orig_j][orig_i] = max(base_heur[orig_j][orig_i], boost_factor / sub_dist[orig_j][orig_i])
            
        else:
            # Held-Karp
            # dp[mask][u] = min cost to start at 0, visit all nodes in mask (including 0 and u), ending at u
            # mask must include 0 and u
            size = 1 << m
            dp = np.full((size, m), np.inf)
            parent = np.full((size, m), -1)
            
            # Start at 0
            dp[1 << 0][0] = 0
            
            for mask in range(size):
                if not (mask & 1): # Must include start node 0
                    continue
                for u in range(m):
                    if not (mask & (1 << u)):
                        continue
                    if dp[mask][u] == np.inf:
                        continue
                    for v in range(m):
                        if mask & (1 << v):
                            continue
                        new_mask = mask | (1 << v)
                        cost = dp[mask][u] + sub_dist[u][v]
                        if cost < dp[new_mask][v]:
                            dp[new_mask][v] = cost
                            parent[new_mask][v] = u
                            
            # Find min cost to return to 0 from full mask
            full_mask = size - 1
            best_last = -1
            best_cost = np.inf
            for u in range(1, m):
                cost = dp[full_mask][u] + sub_dist[u][0]
                if cost < best_cost:
                    best_cost = cost
                    best_last = u
                    
            # Reconstruct path
            if best_last != -1:
                path = []
                mask = full_mask
                u = best_last
                while u != -1:
                    path.append(u)
                    prev = parent[mask][u]
                    mask = mask ^ (1 << u)
                    u = prev
                # path is reversed: [last, ..., 1, 0]
                path.reverse() # [0, 1, ..., last]
                
                # Record edges
                for k in range(len(path) - 1):
                    u_idx = path[k]
                    v_idx = path[k+1]
                    orig_i = sub_nodes[u_idx]
                    orig_j = sub_nodes[v_idx]
                    # Boost heuristic
                    val = boost_factor / (sub_dist[orig_i][orig_j] + 1e-9)
                    base_heur[orig_i][orig_j] = max(base_heur[orig_i][orig_j], val)
                    base_heur[orig_j][orig_i] = max(base_heur[orig_j][orig_i], val)
                
                # Edge from last to depot
                u_idx = path[-1]
                v_idx = 0
                orig_i = sub_nodes[u_idx]
                orig_j = sub_nodes[v_idx]
                val = boost_factor / (sub_dist[orig_i][orig_j] + 1e-9)
                base_heur[orig_i][orig_j] = max(base_heur[orig_i][orig_j], val)
                base_heur[orig_j][orig_i] = max(base_heur[orig_j][orig_i], val)

    # Ensure no zeros or negatives, enforce minimum value
    base_heur[base_heur <= 0] = 1e-9
    
    return base_heur
# EVOLVE-BLOCK-END
