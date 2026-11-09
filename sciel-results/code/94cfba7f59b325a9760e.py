import numpy as np
import time

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Updates the edge distance matrix by adding penalties to edges
    frequently used in previous local optima (Guided Local Search).
    Combines regret-based adaptive local penalties with global usage damping.
    """
    start_time = time.time()
    deadline = start_time + 2.0  # Safety margin

    n = edge_distance.shape[0]
    if n == 0:
        return edge_distance.copy()

    # Extract edges of the current local optimum tour
    tour = np.asarray(local_opt_tour)
    if len(tour) < 2:
        return edge_distance.copy()

    # Get edge indices
    u = tour
    v = np.roll(tour, -1)

    # Get true distances for current tour edges
    current_edge_lengths = edge_distance[u, v]
    tour_len = np.sum(current_edge_lengths)
    avg_edge_len = np.mean(current_edge_lengths)

    # Fallback for degenerate cases
    if not np.isfinite(tour_len) or tour_len == 0:
        tour_len = np.sum(edge_distance) / n
        avg_edge_len = tour_len / n
    if not np.isfinite(avg_edge_len) or avg_edge_len == 0:
        avg_edge_len = 0.1

    max_dist = np.max(edge_distance)
    if not np.isfinite(max_dist) or max_dist == 0:
        max_dist = 1.0

    # --- Regret-based adaptive local penalty ---
    # For each tour edge, compute regret: how much worse the best alternative is
    # compared to the current edge. Edges with low regret (cheap to replace)
    # get stronger penalties, guiding 2-opt to remove them first.
    
    # Compute for each node in the tour, the distance to its two neighbors
    # and the best alternative (non-tour) neighbor
    # For edge (u_i, v_i), regret = min_{j not in {u_{i-1}, u_{i+1}}} d(u_i, j) - d(u_i, v_i)
    # But we need to be careful: the "alternative" should connect the two sides of the tour
    
    # Simpler regret: for each edge (a, b) in the tour, compute
    # the shortest edge from a to any node NOT in the tour neighborhood
    # Actually, a cleaner definition: for edge (a,b), regret is the difference
    # between the current edge length and the best alternative edge that could
    # replace it while maintaining a valid tour structure.
    
    # Practical approximation: for each node, find the shortest edge to a non-adjacent node
    # and compare with the current tour edges incident to that node.
    
    # For each node i, compute the best alternative connection
    # that is not one of its two tour neighbors
    node_min_alt = np.zeros(n)
    for i in range(n):
        node = tour[i]
        prev_node = tour[(i - 1) % n]
        next_node = tour[(i + 1) % n]
        # Find the shortest edge from node to any other node that is not prev or next
        dists = edge_distance[node, :].copy()
        dists[prev_node] = np.inf
        dists[next_node] = np.inf
        dists[node] = np.inf
        if np.all(np.isinf(dists)):
            node_min_alt[i] = avg_edge_len
        else:
            node_min_alt[i] = np.min(dists)
    
    # For each tour edge (u_i, v_i), the regret is related to how easily
    # the endpoints can be reconnected to other nodes
    # Regret for edge (a, b) = (min_alt_a + min_alt_b) / 2 - d(a, b)
    # If regret is negative or small, the edge is "expensive" to keep
    # (i.e., alternatives are cheaper), so we should penalize it less
    # If regret is large positive, alternatives are much cheaper, penalize more
    
    # Actually, let me rethink: we want to penalize edges that are EASY to replace.
    # An edge is easy to replace if there are good alternatives.
    # So: ease_of_replacement = d(a,b) - (best_alt_a + best_alt_b)/2
    # If ease is high (d(a,b) is much larger than alternatives), penalize strongly.
    # If ease is low (d(a,b) is close to alternatives), penalize weakly.
    
    edge_ease = np.zeros(n)
    for i in range(n):
        a = u[i]
        b = v[i]
        # Find the index of a and b in the tour
        # node_min_alt is indexed by tour position, not node id
        # We need to map node ids to tour positions
        # This is expensive; let's use a different approach
        
        # For edge (a, b), the best alternative for a is the shortest edge
        # from a to any node that is not b and not a's other tour neighbor
        # Similarly for b.
        # But this is O(n^2) per edge. Let's precompute.
        pass
    
    # Precompute: for each node, the shortest edge to any node
    # and the second shortest, excluding self
    # Then for edge (a,b), the best alternative for a (excluding b) is:
    #   if the shortest edge from a is to b, use the second shortest
    #   else use the shortest
    
    # Actually, simpler: just compute for each node the shortest edge to any
    # non-tour-neighbor. We already have node_min_alt indexed by tour position.
    
    # Map node id -> tour position
    node_to_pos = np.zeros(n, dtype=int)
    for i in range(n):
        node_to_pos[tour[i]] = i
    
    edge_ease = np.zeros(n)
    for i in range(n):
        a = u[i]
        b = v[i]
        pos_a = node_to_pos[a]
        pos_b = node_to_pos[b]
        # The best alternative for a (not going to b) is node_min_alt[pos_a]
        # but node_min_alt[pos_a] already excludes the two tour neighbors of a
        # which are tour[pos_a - 1] and tour[pos_a + 1]
        # One of these is b. The other is the other tour neighbor.
        # So node_min_alt[pos_a] is the best edge from a to a non-tour-neighbor.
        # Similarly for b.
        alt_a = node_min_alt[pos_a]
        alt_b = node_min_alt[pos_b]
        # Ease: how much cheaper the alternatives are compared to the current edge
        # If d(a,b) is large relative to alternatives, ease is high
        edge_ease[i] = current_edge_lengths[i] - (alt_a + alt_b) / 2.0
    
    # Normalize ease to [0, 1]
    max_ease = np.max(edge_ease)
    min_ease = np.min(edge_ease)
    if max_ease > min_ease:
        ease_norm = (edge_ease - min_ease) / (max_ease - min_ease)
    else:
        ease_norm = np.ones(n) / n
    
    # --- Global usage-based penalty with damping ---
    # Use sqrt damping to avoid over-penalizing heavily used edges
    decayed_usage = np.sqrt(edge_n_used + 1.0) - 1.0
    
    # Base penalty from accumulated usage
    alpha_global = 0.05
    global_penalty = (alpha_global * avg_edge_len) * decayed_usage
    
    # --- Local penalty: regret-based adaptive ---
    # Target penalty: scale so that the current tour becomes non-optimal
    margin = 0.08
    target_penalty_per_edge = (tour_len * margin) / n
    
    # Scale local penalty by ease of replacement
    # Edges that are easy to replace (high ease) get stronger penalties
    local_penalty_scale = 0.5 + 0.5 * ease_norm  # Range [0.5, 1.0]
    
    # Also scale by usage: edges used more often get slightly more penalty
    edge_usage_in_tour = edge_n_used[u, v]
    max_usage = np.max(edge_n_used) if np.max(edge_n_used) > 0 else 1.0
    usage_factor = 1.0 + 0.3 * (edge_usage_in_tour / max_usage)
    
    local_penalty = target_penalty_per_edge * local_penalty_scale * usage_factor
    
    # --- Combine penalties ---
    penalized_dist = edge_distance + global_penalty
    
    # Add local penalties to tour edges
    penalized_dist[u, v] += local_penalty
    penalized_dist[v, u] += local_penalty
    
    # Ensure the diagonal is 0
    np.fill_diagonal(penalized_dist, 0.0)
    
    # Ensure symmetry
    penalized_dist = (penalized_dist + penalized_dist.T) / 2.0
    
    # Check for finiteness
    if not np.all(np.isfinite(penalized_dist)):
        penalized_dist = np.nan_to_num(penalized_dist, nan=0.0, posinf=max_dist * 10, neginf=0.0)
    
    # Time check
    if time.time() > deadline:
        return edge_distance.copy()
    
    return penalized_dist
