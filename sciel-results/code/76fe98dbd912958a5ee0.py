import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s hard kill

# Cache for coordinates derived from distance matrix (Euclidean assumption)
_COORD_CACHE = {}


def _get_coordinates(distance_matrix):
    """
    Recover 2D coordinates from a Euclidean distance matrix using MDS.
    Returns (coords, success).
    """
    key = id(distance_matrix)
    if key in _COORD_CACHE:
        return _COORD_CACHE[key]
    
    n = distance_matrix.shape[0]
    try:
        # Use classical MDS to recover coordinates
        D2 = distance_matrix ** 2
        # Centering matrix
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ D2 @ J
        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        # Take top 2 eigenvalues
        idx = np.argsort(eigenvalues)[::-1][:2]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        # Coordinates
        coords = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))
        _COORD_CACHE[key] = (coords, True)
        return coords, True
    except Exception:
        _COORD_CACHE[key] = (None, False)
        return None, False


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node for a TSP tour construction.
    
    Uses a combination of:
    1. Distance from current node (local proximity)
    2. Distance to destination (global completion pressure)
    3. A lightweight 1-step or 2-step lookahead for small remaining sets
    4. Connectivity penalty for isolated nodes
    
    The weights adapt based on how many nodes remain unvisited.
    """
    # Time guard - if we're running out of time, fall back to greedy
    if time.time() - _START_TIME > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        # For two nodes left, pick the one that minimizes total edge cost to close the tour
        u, v = unvisited_nodes
        cost_u = distance_matrix[current_node, u] + distance_matrix[u, v] + distance_matrix[v, destination_node]
        cost_v = distance_matrix[current_node, v] + distance_matrix[v, u] + distance_matrix[u, destination_node]
        return int(unvisited_nodes[0] if cost_u <= cost_v else unvisited_nodes[1])
    if n_unvisited == 3:
        # For three nodes, evaluate all 6 permutations
        u, v, w = unvisited_nodes
        perms = [
            (u, v, w), (u, w, v),
            (v, u, w), (v, w, u),
            (w, u, v), (w, v, u)
        ]
        best_cost = float('inf')
        best_first = u
        for p in perms:
            c = (distance_matrix[current_node, p[0]] + 
                 distance_matrix[p[0], p[1]] + 
                 distance_matrix[p[1], p[2]] + 
                 distance_matrix[p[2], destination_node])
            if c < best_cost:
                best_cost = c
                best_first = p[0]
        return int(best_first)
    
    # Adaptive weight: more emphasis on destination as we get closer to finishing
    # alpha: weight on distance to destination
    # At n_unvisited = 50, alpha ~ 0.15; at n_unvisited = 5, alpha ~ 0.6
    progress = 1.0 - n_unvisited / 50.0
    alpha = 0.15 + 0.45 * progress
    alpha = max(alpha, 0.1)
    alpha = min(alpha, 0.6)
    
    # Distances from current to all unvisited
    d_curr = distance_matrix[current_node, unvisited_nodes]
    
    # Distances from each unvisited to destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # For small remaining sets, do a proper 2-step lookahead
    if n_unvisited <= 15:
        # For each candidate, estimate: d(current, cand) + min over next of (d(cand, next) + d(next, dest) * beta)
        # where beta is a small weight for the closing edge
        # This is a 2-step lookahead
        d_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        
        lookahead_costs = np.zeros(n_unvisited)
        for i in range(n_unvisited):
            cand = unvisited_nodes[i]
            # Cost to reach candidate
            base_cost = d_curr[i]
            # For each possible next node j (j != i), compute d(cand, j) + small weight * d(j, dest)
            if n_unvisited > 1:
                others = np.delete(unvisited_nodes, i)
                d_cand_to_others = distance_matrix[cand, others]
                d_others_to_dest = distance_matrix[others, destination_node]
                # Weighted combination for the lookahead step
                # We want to minimize the continuation, so take min
                continuation = np.min(d_cand_to_others + 0.3 * d_others_to_dest)
                lookahead_costs[i] = base_cost + continuation
            else:
                lookahead_costs[i] = base_cost
        
        # Normalize lookahead to comparable scale with d_curr
        min_lh = np.min(lookahead_costs)
        max_lh = np.max(lookahead_costs)
        if max_lh > min_lh:
            lookahead_norm = (lookahead_costs - min_lh) / (max_lh - min_lh)
        else:
            lookahead_norm = np.zeros(n_unvisited)
        
        # Combine with appropriate weights
        # Primary signal is the lookahead cost itself, normalized
        # We blend it with the direct distance to preserve local structure
        gamma = 0.5 + 0.3 * progress  # More lookahead weight as we finish
        scores = d_curr + alpha * d_dest + gamma * lookahead_norm * (max_lh - min_lh)
        
    else:
        # For larger n_unvisited, use a faster heuristic
        # Compute nearest neighbor distance from each unvisited to other unvisited
        # (excluding self) as a connectivity measure
        d_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        np.fill_diagonal(d_sub, np.inf)
        d_connect = np.min(d_sub, axis=1)
        
        # Also compute a 1-step lookahead: for each candidate, 
        # d(current, cand) + min_j != cand (d(cand, j))
        # This captures "can I go to this node and still easily reach others?"
        one_step_lh = d_curr.copy()
        for i in range(n_unvisited):
            cand = unvisited_nodes[i]
            others_mask = np.ones(n_unvisited, dtype=bool)
            others_mask[i] = False
            d_cand_to_others = d_sub[i, others_mask]
            if len(d_cand_to_others) > 0:
                one_step_lh[i] += np.min(d_cand_to_others)
        
        min_1lh = np.min(one_step_lh)
        max_1lh = np.max(one_step_lh)
        if max_1lh > min_1lh:
            one_step_norm = (one_step_lh - min_1lh) / (max_1lh - min_1lh)
        else:
            one_step_norm = np.zeros(n_unvisited)
        
        # Normalize connectivity
        min_c = np.min(d_connect)
        max_c = np.max(d_connect)
        if max_c > min_c:
            connect_norm = (d_connect - min_c) / (max_c - min_c)
        else:
            connect_norm = np.zeros(n_unvisited)
        
        # Combine all signals
        beta_connect = 0.1
        gamma_1step = 0.2 + 0.2 * progress
        
        scores = d_curr + alpha * d_dest + beta_connect * connect_norm * (max_c - min_c) + gamma_1step * one_step_norm * (max_1lh - min_1lh)
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
