"""Traveling salesman constructive heuristic with regret-based savings and soft-min smoothing."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Strategy: Regret-based savings heuristic with soft-min smoothing.
#
# Instead of pure distance + connectivity, we use a *regret* score that captures
# the opportunity cost of NOT visiting a candidate now. For each candidate u:
#   - regret(u) = (second_best_incoming - best_incoming) where "incoming" edges
#     are from current_node or from other unvisited nodes.
#   - This means: if u has only one cheap way to be reached (from current),
#     skipping it now forces a much more expensive connection later.
#
# We combine:
#   score(u) = d(current, u) + alpha * regret(u) + beta * softmin_outgoing(u)
#
# where softmin_outgoing(u) is a smoothed version of the min distance from u to
# other unvisited nodes, providing a gentle lookahead without hard commitment.
#
# The alpha weight adapts: higher early in construction (more exploratory,
# penalize high-regret skips), lower late (more greedy).

def _softmin(values, temperature=0.5):
    """Compute soft minimum: -T * log(sum(exp(-x/T))). Falls back to hard min if T->0."""
    if temperature <= 1e-10:
        return np.min(values)
    # Numerically stable softmin
    shifted = -values / temperature
    shifted -= np.max(shifted)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return -temperature * (log_sum + np.max(shifted))


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using regret-based savings with soft-min smoothing.
    
    Regret captures the opportunity cost of skipping a candidate: if a node has
    only one cheap connection (from current), skipping it forces expensive
    reconnections later. Soft-min provides smooth lookahead.
    """
    if len(unvisited_nodes) == 0:
        return -1
        
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    n_remaining = len(unvisited_nodes)
    t_start = time.time()
    DEADLINE = 145.0
    
    # 1. Distance from current to each candidate
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Sub-matrix of distances between unvisited nodes
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    sub_matrix_no_diag = sub_matrix.copy()
    np.fill_diagonal(sub_matrix_no_diag, np.inf)
    
    # 3. Regret computation:
    # For each candidate u, consider all "incoming" edges: from current_node and
    # from each other unvisited node. The regret is (2nd best - best).
    # If current is the unique best incoming, skipping u is very costly.
    
    # Incoming distances for each u: [d(current,u), d(u1,u), d(u2,u), ...]
    # We need the two smallest incoming distances for each u.
    
    # d(current, u) is already dists_from_current
    # d(v, u) for v in unvisited is sub_matrix_no_diag[:, u] (column u, excluding u itself)
    
    # Build full incoming matrix: rows = unvisited candidates, cols = [current] + other unvisited
    # Actually, for each candidate u (row index i in unvisited_nodes):
    #   incoming[i] = [d(current, u_i)] + [d(u_j, u_i) for j != i in unvisited]
    
    # d(current, u_i) = dists_from_current[i]
    # d(u_j, u_i) = sub_matrix_no_diag[j, i] (but sub_matrix_no_diag has inf on diagonal)
    
    # So for each i, incoming values are: dists_from_current[i] and sub_matrix_no_diag[:, i]
    # (the latter includes inf at position i, which is fine)
    
    # Stack them: shape (n_remaining, n_remaining + 1)
    incoming = np.column_stack([dists_from_current, sub_matrix_no_diag.T])
    # incoming[i, 0] = d(current, u_i)
    # incoming[i, j+1] = d(u_j, u_i) for j in range(n_remaining)
    
    # For each row, find the two smallest values
    # Use argsort and take first two
    sorted_incoming = np.sort(incoming, axis=1)
    best_incoming = sorted_incoming[:, 0]
    second_best_incoming = sorted_incoming[:, 1]
    
    regret = second_best_incoming - best_incoming
    
    # 4. Soft-min outgoing: for each u, softmin of distances from u to other unvisited
    # sub_matrix_no_diag[i, :] gives distances from u_i to all other unvisited
    # (with inf at position i)
    temperature = 0.3  # moderate smoothing
    softmin_outgoing = np.array([_softmin(sub_matrix_no_diag[i, :], temperature) for i in range(n_remaining)])
    
    # 5. Adaptive alpha: higher early, lower late
    # Progress: 0 at start, 1 near end
    total_nodes = distance_matrix.shape[0]
    progress = 1.0 - (n_remaining / max(total_nodes, 1))
    # alpha starts at 0.8, decays to 0.2
    alpha = 0.8 - 0.6 * progress
    # beta starts at 0.3, decays to 0.1
    beta = 0.3 - 0.2 * progress
    
    # 6. Combine scores
    scores = dists_from_current + alpha * regret + beta * softmin_outgoing
    
    # Check deadline
    if time.time() > DEADLINE:
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
