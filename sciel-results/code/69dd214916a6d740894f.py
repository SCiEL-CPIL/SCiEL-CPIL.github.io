"""Savings-based TSP constructive heuristic: prioritize nodes with high insertion savings."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 135.0  # safety margin within 150s wall-clock kill


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next city using a savings-inspired heuristic.
    
    For each candidate j, compute the 'savings' of inserting j into the tour
    between current_node and destination_node:
      savings(j) = d(current, destination) - d(current, j) - d(j, destination)
    
    We prefer candidates with HIGH savings (i.e., inserting j avoids a long
    direct edge). Additionally, we use a small nearest-neighbor term for
    stability when savings are similar.
    
    As the tour nears completion, we blend in pure destination-proximity to
    ensure the final segment is short.
    
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

    # Base savings: how much we save by going through j instead of direct
    d_direct = d[current_node, destination_node]
    d_curr = d[current_node, unvisited]  # shape (m,)
    d_dest = d[unvisited, destination_node]  # shape (m,)
    
    # Savings: positive means j is a good intermediate (avoids long edge)
    savings = d_direct - d_curr - d_dest  # shape (m,)
    
    # Normalize savings to [0, 1] for blending
    s_min = np.min(savings)
    s_max = np.max(savings)
    if s_max > s_min:
        savings_norm = (savings - s_min) / (s_max - s_min)
    else:
        savings_norm = np.zeros_like(savings)
    
    # Nearest-neighbor score (lower is better)
    nn_score = d_curr / (np.max(d_curr) + 1e-9)
    
    # Destination proximity score (lower is better)
    dest_score = d_dest / (np.max(d_dest) + 1e-9)
    
    # Adaptive weights: as we near the end, prioritize destination proximity
    frac = n_unvisited / max(n_total - 1, 1)
    # Early: weight savings heavily; Late: weight destination proximity heavily
    w_savings = 0.7 * frac ** 0.5
    w_nn = 0.3 * frac ** 0.5
    w_dest = 1.0 - w_savings - w_nn
    if w_dest < 0:
        w_dest = 0.0
        # Renormalize
        total = w_savings + w_nn
        if total > 0:
            w_savings /= total
            w_nn /= total
        else:
            w_savings, w_nn, w_dest = 0.0, 0.0, 1.0
    
    # Composite score: minimize (lower is better)
    # High savings_norm is good, so we subtract it
    scores = w_nn * nn_score + w_dest * dest_score - w_savings * savings_norm
    
    best_idx = int(np.argmin(scores))
    return int(unvisited[best_idx])
# EVOLVE-BLOCK-END
