"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _two_opt_fast(tour, dist, max_passes=5):
    """Improve a tour using vectorized 2-opt."""
    n = len(tour)
    tour = np.array(tour, dtype=int)
    
    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            a = tour[i]
            b = tour[i + 1]
            # For all j > i+1, compute the 2-opt delta
            c = tour[i + 2:]
            d = tour[i + 3:]
            d = np.append(d, tour[0])  # wrap around
            c = c[:len(d)]
            # Actually need to be more careful with the wrap
            c_arr = tour[i + 2:]
            d_arr = np.empty_like(c_arr)
            d_arr[:-1] = tour[i + 3:]
            d_arr[-1] = tour[0]
            
            old_cost = dist[a, b] + dist[c_arr, d_arr]
            new_cost = dist[a, c_arr] + dist[b, d_arr]
            deltas = new_cost - old_cost
            best_j = np.argmin(deltas)
            if deltas[best_j] < -1e-12:
                j = i + 2 + best_j
                tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                improved = True
        if not improved:
            break
    
    return tour.tolist()


def _two_opt(tour, dist):
    """Improve a tour using 2-opt with early termination."""
    n = len(tour)
    improved = True
    passes = 0
    max_passes = 10
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                old_cost = dist[a, b] + dist[c, d]
                new_cost = dist[a, c] + dist[b, d]
                if new_cost < old_cost - 1e-12:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
    return tour


def _or_opt(tour, dist, segment_length=2):
    """Improve a tour using Or-opt (move a small segment)."""
    n = len(tour)
    improved = True
    passes = 0
    max_passes = 5
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n):
            # Try moving segment of length segment_length starting at i
            seg = tour[i:i + segment_length]
            if len(seg) < segment_length:
                continue
            # Remove segment
            remaining = tour[:i] + tour[i + segment_length:]
            # Try inserting at each position
            best_pos = -1
            best_delta = 0
            for pos in range(len(remaining)):
                # Cost before: edges (remaining[pos-1], remaining[pos]) 
                # and (tour[i-1], tour[i]) + (tour[i+segment_length-1], tour[i+segment_length])
                # This is complex, so we just check if insertion improves
                insert_before = remaining[pos]
                insert_after = remaining[(pos + 1) % len(remaining)]
                prev_node = remaining[pos - 1]
                
                old_cost = dist[prev_node, insert_before] + dist[insert_before, insert_after]
                new_cost = dist[prev_node, seg[0]] + dist[seg[-1], insert_after]
                delta = new_cost - old_cost
                if delta < best_delta - 1e-12:
                    best_delta = delta
                    best_pos = pos
            
            if best_pos >= 0:
                # Perform the move
                remaining = tour[:i] + tour[i + segment_length:]
                new_tour = remaining[:best_pos] + seg + remaining[best_pos:]
                tour = new_tour
                improved = True
    return tour


def _nearest_neighbor_tour(dist, start=0):
    """Generate a tour using nearest neighbor heuristic."""
    n = dist.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    tour = [start]
    current = start
    for _ in range(n - 1):
        dists = dist[current, :].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    return tour


def _greedy_tour(dist, start=0):
    """Generate a tour using greedy edge insertion."""
    n = dist.shape[0]
    # Start with the node closest to start
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    tour = [start]
    current = start
    
    for _ in range(n - 1):
        dists = dist[current, :].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    return tour


def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s limit
    n = distance_matrix.shape[0]
    
    # Generate multiple diverse reference tours and optimize them
    reference_tours = []
    
    # Strategy 1: Nearest neighbor from node 0
    nn_tour = _nearest_neighbor_tour(distance_matrix, 0)
    optimized = _two_opt(nn_tour, distance_matrix)
    reference_tours.append(optimized)
    
    if time.time() > deadline - 30:
        pass  # Continue with what we have
    
    # Strategy 2: Nearest neighbor from a few other starts
    for start in [1, n // 2, n // 3]:
        if time.time() > deadline - 20:
            break
        nn_tour = _nearest_neighbor_tour(distance_matrix, start)
        optimized = _two_opt(nn_tour, distance_matrix)
        reference_tours.append(optimized)
    
    # Strategy 3: Greedy from different starts
    for start in [0, n // 2]:
        if time.time() > deadline - 10:
            break
        g_tour = _greedy_tour(distance_matrix, start)
        optimized = _two_opt(g_tour, distance_matrix)
        reference_tours.append(optimized)
    
    # Apply Or-opt to the best tour found so far
    if reference_tours and time.time() < deadline - 5:
        best_tour = min(reference_tours, key=lambda t: sum(
            distance_matrix[t[i], t[(i + 1) % n]] for i in range(n)
        ))
        or_optimized = _or_opt(best_tour[:], distance_matrix, segment_length=2)
        reference_tours.append(or_optimized)
    
    # Build edge attractiveness matrix using multiple signals
    # Signal 1: Edge frequency in reference tours (weighted by tour quality)
    # Signal 2: Inverse distance (shorter edges are more attractive)
    # Signal 3: Edge centrality (edges that appear in many good tours)
    
    # Compute tour lengths
    tour_lengths = []
    for tour in reference_tours:
        tour_len = sum(distance_matrix[tour[i], tour[(i + 1) % n]] for i in range(n))
        tour_lengths.append(tour_len)
    
    # Normalize tour lengths (shorter is better)
    min_len = min(tour_lengths)
    max_len = max(tour_lengths)
    if max_len > min_len:
        tour_weights = [(max_len - l) / (max_len - min_len + 1e-12) + 0.1 for l in tour_lengths]
    else:
        tour_weights = [1.0] * len(tour_lengths)
    
    # Edge attractiveness from tour frequency
    edge_attractiveness = np.zeros((n, n))
    
    for tour, weight in zip(reference_tours, tour_weights):
        for i in range(n):
            u = tour[i]
            v = tour[(i + 1) % n]
            edge_attractiveness[u, v] += weight
            edge_attractiveness[v, u] += weight * 0.3  # Asymmetric: forward direction stronger
    
    # Inverse distance signal (with smoothing)
    inv_dist = 1.0 / (distance_matrix + 1e-9)
    # Normalize inverse distance
    inv_dist_max = np.max(inv_dist)
    inv_dist_normalized = inv_dist / (inv_dist_max + 1e-12)
    
    # Combine signals
    # Scale edge attractiveness to comparable range
    ea_max = np.max(edge_attractiveness)
    if ea_max > 0:
        edge_attractiveness_scaled = edge_attractiveness / ea_max
    else:
        edge_attractiveness_scaled = edge_attractiveness
    
    # Combined heuristic: blend of tour-based and distance-based
    blend_alpha = 0.6  # Weight for tour-based signal
    combined = blend_alpha * edge_attractiveness_scaled + (1 - blend_alpha) * inv_dist_normalized
    
    # Set diagonal to 0
    np.fill_diagonal(combined, 0.0)
    
    # Apply power law to amplify differences
    mask = ~np.eye(n, dtype=bool)
    min_val = np.min(combined[mask])
    max_val = np.max(combined[mask])
    
    if max_val > min_val + 1e-12:
        normalized = (combined - min_val) / (max_val - min_val + 1e-12)
        # Power > 1 amplifies top edges
        powered = normalized ** 1.5
        edge_attractiveness_final = 1.0 + 20.0 * powered
    else:
        edge_attractiveness_final = 1.0 + 20.0 * normalized
    
    # Ensure diagonal is 0
    np.fill_diagonal(edge_attractiveness_final, 0.0)
    
    # Ensure all off-diagonal entries are positive
    edge_attractiveness_final[mask] = np.maximum(edge_attractiveness_final[mask], 1e-9)
    
    # Final sanity check
    if not np.all(np.isfinite(edge_attractiveness_final)):
        # Fallback to simple inverse distance
        fallback = 1.0 / (distance_matrix + 1e-9)
        np.fill_diagonal(fallback, 0.0)
        fallback[mask] = np.maximum(fallback[mask], 1e-9)
        return fallback
    
    return edge_attractiveness_final
# EVOLVE-BLOCK-END
