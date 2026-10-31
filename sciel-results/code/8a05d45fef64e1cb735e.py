"""Geometric alignment-based TSP constructive heuristic: favor nodes on the path to destination."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 135.0  # safety margin within 150s wall-clock kill


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next city using a geometric alignment heuristic.
    
    The key insight for Euclidean TSP is that optimal tours tend to follow
    smooth, non-crossing paths. A candidate j is "aligned" with the direction
    from current to destination if the detour cost is small:
      detour(j) = d(current, j) + d(j, destination) - d(current, destination)
    
    We minimize a composite score that favors:
      1. Low detour (good alignment with the path to destination)
      2. Reasonable proximity to current node (avoid jumping too far)
    
    As the tour nears completion, we increase the weight on destination proximity
    to ensure the final edge back to the start is short.
    
    Parameters
    ----------
    current_node : int
    destination_node : int
    unvisited_nodes : np.ndarray
    distance_matrix : np.ndarray
    Returns
    -------
    int : index of the next node to visit
    """
    unvisited = np.asarray(unvisited_nodes)
    n_unvisited = len(unvisited)
    d = distance_matrix
    n_total = d.shape[0]

    if n_unvisited == 0:
        return int(current_node)
    if n_unvisited == 1:
        return int(unvisited[0])

    # Deadline guard: fall back to pure nearest-neighbor
    if time.time() > _DEADLINE - 2.0:
        d_curr = d[current_node, unvisited]
        return int(unvisited[int(np.argmin(d_curr))])

    # Compute distances
    d_direct = d[current_node, destination_node]
    d_curr = d[current_node, unvisited]  # shape (m,)
    d_dest = d[unvisited, destination_node]  # shape (m,)
    
    # Detour cost: how much extra distance we add by going through j
    # Negative means j is "beyond" the direct path (rare in Euclidean)
    detour = d_curr + d_dest - d_direct  # shape (m,)
    
    # Normalize detour to [0, 1] (lower is better)
    det_min = np.min(detour)
    det_max = np.max(detour)
    if det_max > det_min:
        detour_norm = (detour - det_min) / (det_max - det_min)
    else:
        detour_norm = np.zeros_like(detour)
    
    # Proximity to current node (lower is better)
    prox_curr = d_curr / (np.max(d_curr) + 1e-9)
    
    # Proximity to destination (lower is better)
    prox_dest = d_dest / (np.max(d_dest) + 1e-9)
    
    # Adaptive weights based on progress through the tour
    frac = n_unvisited / max(n_total - 1, 1)
    # Early: weight alignment heavily; Late: weight destination proximity heavily
    w_detour = 0.6 * frac ** 0.5
    w_prox_curr = 0.2 * frac ** 0.5
    w_prox_dest = 1.0 - w_detour - w_prox_curr
    if w_prox_dest < 0:
        w_prox_dest = 0.0
        total = w_detour + w_prox_curr
        if total > 0:
            w_detour /= total
            w_prox_curr /= total
        else:
            w_detour, w_prox_curr, w_prox_dest = 0.0, 0.0, 1.0
    
    # Composite score: minimize (lower is better)
    scores = w_detour * detour_norm + w_prox_curr * prox_curr + w_prox_dest * prox_dest
    
    best_idx = int(np.argmin(scores))
    return int(unvisited[best_idx])
# EVOLVE-BLOCK-END
