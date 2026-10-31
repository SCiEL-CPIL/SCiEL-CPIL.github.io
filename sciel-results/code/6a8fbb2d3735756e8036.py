"""TSP constructive heuristic: force-directed alignment with MST-leaf awareness."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TOTAL_LIMIT = 150.0
_MARGIN = 5.0

def _is_deadline_passed():
    """Check if we are close to the time limit."""
    return (time.time() - _start_time) > (_TOTAL_LIMIT - _MARGIN)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Strategy: Force-directed geometric alignment with MST-leaf peeling.
    
    1. Reconstruct 2D coordinates from the distance matrix using PCA (SVD).
    2. Compute a "force" on each unvisited node:
       - Attraction toward current node (nearest-neighbor pull)
       - Attraction toward destination (tour-closing pull, weighted by progress)
       - Repulsion from the centroid of visited nodes (encourages spreading out)
    3. Score each candidate by the alignment of its net force vector with the
       direction from current to destination, plus a small MST-degree penalty
       to prefer "leaf" nodes in the MST of unvisited nodes.
    4. Select the node with the best (lowest) combined score.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Deadline check: fall back to fast nearest-neighbor
    if _is_deadline_passed():
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
    
    total_nodes = distance_matrix.shape[0]
    
    # Progress: 0 at start, 1 when only 1 node remains
    if total_nodes > 1:
        progress = 1.0 - (n_unvisited / total_nodes)
    else:
        progress = 1.0
    
    # Precompute distances
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # --- Reconstruct 2D coordinates via PCA on the distance matrix ---
    # Use double-centering of the squared distance matrix for the full set
    # For efficiency, we compute once per call. Since n is small (50), this is fine.
    try:
        D = distance_matrix
        D_sq = D ** 2
        # Double centering: B = -0.5 * J * D_sq * J
        n_total = D.shape[0]
        J = np.eye(n_total) - np.ones((n_total, n_total)) / n_total
        B = -0.5 * J @ D_sq @ J
        # Eigenvalue decomposition (or SVD) to get coordinates
        eigvals, eigvecs = np.linalg.eigh(B)
        # Take the two largest eigenvalues
        idx = np.argsort(eigvals)[::-1][:2]
        eigvals_sel = eigvals[idx]
        eigvecs_sel = eigvecs[:, idx]
        # Coordinates: eigvecs_sel * sqrt(eigvals_sel)
        coords = eigvecs_sel * np.sqrt(np.maximum(eigvals_sel, 0))
    except Exception:
        # Fallback: use a simple identity-like coordinate system
        coords = np.eye(total_nodes)
    
    # Extract coordinates for relevant nodes
    curr_coord = coords[current_node]
    dest_coord = coords[destination_node]
    unvisited_coords = coords[unvisited_nodes]
    
    # Centroid of visited nodes (approximate: all nodes except unvisited)
    visited_mask = np.ones(total_nodes, dtype=bool)
    visited_mask[unvisited_nodes] = False
    if np.any(visited_mask):
        visited_coords = coords[visited_mask]
        centroid_visited = np.mean(visited_coords, axis=0)
    else:
        centroid_visited = curr_coord
    
    # Direction from current to destination
    dir_to_dest = dest_coord - curr_coord
    dir_to_dest_norm = np.linalg.norm(dir_to_dest)
    if dir_to_dest_norm > 1e-10:
        dir_to_dest_unit = dir_to_dest / dir_to_dest_norm
    else:
        dir_to_dest_unit = np.zeros(2)
    
    # --- Compute force-directed scores ---
    # For each unvisited node i:
    #   Force = w1 * (curr_coord - coord_i) / |curr_coord - coord_i|  (attraction to current)
    #         + w2 * (dest_coord - coord_i) / |dest_coord - coord_i|  (attraction to dest)
    #         - w3 * (coord_i - centroid_visited) / |coord_i - centroid_visited|  (repulsion from visited centroid)
    
    # Weights adapt with progress
    w1 = 1.0  # Base attraction to current
    w2 = 0.1 + 0.9 * (progress ** 2)  # Increasing attraction to destination
    w3 = 0.3  # Repulsion strength
    
    # Vectorized computation
    # Attraction to current
    vec_to_curr = curr_coord - unvisited_coords  # shape (n_unvisited, 2)
    dists_to_curr = np.linalg.norm(vec_to_curr, axis=1)
    dists_to_curr_safe = np.where(dists_to_curr > 1e-10, dists_to_curr, 1.0)
    force_curr = vec_to_curr / dists_to_curr_safe[:, np.newaxis]
    
    # Attraction to destination
    vec_to_dest = dest_coord - unvisited_coords
    dists_to_dest_coords = np.linalg.norm(vec_to_dest, axis=1)
    dists_to_dest_coords_safe = np.where(dists_to_dest_coords > 1e-10, dists_to_dest_coords, 1.0)
    force_dest = vec_to_dest / dists_to_dest_coords_safe[:, np.newaxis]
    
    # Repulsion from visited centroid
    vec_from_centroid = unvisited_coords - centroid_visited
    dists_from_centroid = np.linalg.norm(vec_from_centroid, axis=1)
    dists_from_centroid_safe = np.where(dists_from_centroid > 1e-10, dists_from_centroid, 1.0)
    force_repel = vec_from_centroid / dists_from_centroid_safe[:, np.newaxis]
    
    # Net force
    net_force = w1 * force_curr + w2 * force_dest - w3 * force_repel
    
    # Alignment score: dot product with direction to destination
    # Higher alignment = better (more aligned with tour direction)
    # We want to MINIMIZE a score, so we use negative alignment
    alignment_scores = -np.sum(net_force * dir_to_dest_unit, axis=1)
    
    # --- MST-leaf penalty ---
    # Compute MST of unvisited nodes and penalize high-degree nodes
    # For small n, use a simple Prim's algorithm
    if n_unvisited > 2:
        d_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        
        # Prim's algorithm to get MST edges
        n_sub = n_unvisited
        dist_to_mst = np.full(n_sub, np.inf)
        dist_to_mst[0] = 0.0
        visited_mst = np.zeros(n_sub, dtype=bool)
        mst_edges = []
        
        for _ in range(n_sub):
            # Find unvisited node with min dist_to_mst
            unvisited_idx = np.where(~visited_mst)[0]
            if len(unvisited_idx) == 0:
                break
            u = unvisited_idx[np.argmin(dist_to_mst[unvisited_idx])]
            visited_mst[u] = True
            
            if _ > 0:  # Not the first node
                # Find which node u was connected to (the one that gave it min dist)
                # Actually, we need to track parent. Let's redo with parent tracking.
                pass
        
        # Redo with parent tracking
        dist_to_mst = np.full(n_sub, np.inf)
        parent = np.full(n_sub, -1, dtype=int)
        dist_to_mst[0] = 0.0
        visited_mst = np.zeros(n_sub, dtype=bool)
        
        for _ in range(n_sub):
            unvisited_idx = np.where(~visited_mst)[0]
            if len(unvisited_idx) == 0:
                break
            u = unvisited_idx[np.argmin(dist_to_mst[unvisited_idx])]
            visited_mst[u] = True
            
            # Update neighbors
            for v in range(n_sub):
                if not visited_mst[v] and d_sub[u, v] < dist_to_mst[v]:
                    dist_to_mst[v] = d_sub[u, v]
                    parent[v] = u
        
        # Compute degrees in MST
        degrees = np.zeros(n_sub, dtype=int)
        for v in range(1, n_sub):  # Skip root
            if parent[v] >= 0:
                degrees[parent[v]] += 1
                degrees[v] += 1
        
        # Median degree for normalization
        median_degree = np.median(degrees) if np.any(degrees > 0) else 1.0
        if median_degree < 1e-10:
            median_degree = 1.0
        
        # Penalty: high degree means more connected, less "leaf-like"
        mst_penalty = (degrees / median_degree - 1.0) * 0.1
    else:
        mst_penalty = np.zeros(n_unvisited)
    
    # --- Immediate distance component (normalized) ---
    mean_dist = np.mean(dists_from_current)
    if mean_dist > 1e-10:
        dist_norm = dists_from_current / mean_dist
    else:
        dist_norm = dists_from_current
    
    # Combine: alignment score + immediate distance + MST penalty
    # The alignment score is already scale-invariant (unit vectors)
    # We weight the immediate distance to ensure we don't pick too-far nodes
    combined_score = 0.5 * alignment_scores + 0.3 * dist_norm + 0.2 * mst_penalty
    
    # Final deadline check
    if _is_deadline_passed():
        return int(unvisited_nodes[np.argmin(dists_from_current)])
    
    best_idx = np.argmin(combined_score)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
