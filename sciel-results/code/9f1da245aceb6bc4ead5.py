"""
Capacity-aware nearest-neighbor reference tour heuristic for ACO CVRP.
Builds a reference tour using capacity-constrained NN, then boosts edges
that appear in or near the reference tour to guide pheromone deposition
toward structurally sound, coherent routes.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix (larger = more attractive) for ACO CVRP.
    
    Strategy: Build a capacity-aware nearest-neighbor reference tour, then
    boost edges that are part of or adjacent to that tour. This provides
    structural guidance that pure inverse-distance heuristics lack, avoiding
    early zig-zag traps and promoting coherent, well-filled routes.
    
    Returns an n x n matrix of the same shape as distance_matrix.
    """
    start_time = time.time()
    deadline = start_time + 280.0

    n = distance_matrix.shape[0]
    if distance_matrix.shape[1] != n:
        return np.full((n, n), 1e-9)

    demands = np.asarray(demands, dtype=np.float64)
    if demands.shape[0] != n:
        demands = np.resize(demands, n)

    # Safe distance matrix
    dist_safe = np.maximum(distance_matrix, 1e-9)
    
    # Base heuristic: inverse distance
    heuristic = 1.0 / dist_safe
    
    if n > 1 and time.time() < deadline:
        # Step 1: Build a capacity-aware nearest-neighbor reference tour
        # This simulates what a good route looks like under capacity constraints
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        current = 0
        load = 0.0
        ref_edges = np.zeros((n, n), dtype=np.float64)
        
        # Track which customers are in which route segment
        route_segments = []  # list of lists of customer indices
        current_segment = []
        
        max_steps = n * 2  # safety bound
        steps = 0
        
        while steps < max_steps:
            steps += 1
            # Find best unvisited customer that fits in remaining capacity
            remaining_cap = capacity - load
            best_j = -1
            best_dist = np.inf
            
            for j in range(1, n):
                if not visited[j] and demands[j] <= remaining_cap:
                    d = distance_matrix[current, j]
                    if d < best_dist:
                        best_dist = d
                        best_j = j
            
            if best_j == -1:
                # No customer fits; return to depot
                ref_edges[current, 0] = 1.0
                if current_segment:
                    route_segments.append(current_segment)
                    current_segment = []
                current = 0
                load = 0.0
                # Check if all visited
                if np.all(visited):
                    break
            else:
                # Move to best_j
                ref_edges[current, best_j] = 1.0
                visited[best_j] = True
                load += demands[best_j]
                current_segment.append(best_j)
                current = best_j
        
        # If we ended not at depot, record the return edge
        if current != 0:
            ref_edges[current, 0] = 1.0
            if current_segment:
                route_segments.append(current_segment)
        
        # Step 2: Build attractiveness from reference tour structure
        # Edges in the reference tour get a strong boost
        # Edges that are "near" reference tour edges (same route segment neighbors) get moderate boost
        
        # First, identify route segment membership for each customer
        segment_id = np.full(n, -1, dtype=np.int64)
        for seg_idx, seg in enumerate(route_segments):
            for cust in seg:
                segment_id[cust] = seg_idx
        
        # Boost matrix: start with 0
        boost = np.zeros((n, n), dtype=np.float64)
        
        # Strong boost for reference tour edges
        for i in range(n):
            for j in range(n):
                if ref_edges[i, j] > 0.5:
                    boost[i, j] = 2.0
        
        # Moderate boost for edges within the same route segment
        # (customers that are close in the reference tour order)
        for seg_idx, seg in enumerate(route_segments):
            m = len(seg)
            for a in range(m):
                for b in range(m):
                    if a != b:
                        # Boost decreases with distance in tour order
                        dist_in_tour = abs(a - b)
                        if dist_in_tour <= 3:
                            strength = 1.0 / (1.0 + 0.5 * dist_in_tour)
                            i, j = seg[a], seg[b]
                            boost[i, j] = max(boost[i, j], strength)
        
        # Also boost depot-to-first-customer and last-customer-to-depot edges
        # (already covered by ref_edges)
        
        # Step 3: Combine base inverse distance with structural boost
        # heuristic[i,j] = base[i,j] * (1 + boost[i,j])
        # This keeps the distance signal but amplifies structurally sound edges
        
        heuristic = heuristic * (1.0 + boost)
        
        # Ensure diagonal is 0 (no self-loops)
        np.fill_diagonal(heuristic, 1e-9)

    # Ensure all entries are strictly positive and finite
    heuristic = np.maximum(heuristic, 1e-9)
    
    if not np.all(np.isfinite(heuristic)):
        heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
        heuristic = np.maximum(heuristic, 1e-9)

    return heuristic
# EVOLVE-BLOCK-END
