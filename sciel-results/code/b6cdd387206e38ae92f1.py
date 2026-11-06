"""Priority function for Maximum Independent Set via greedy decoder.
Strategy: assign scores that favor low-degree vertices while accounting for
the structure of their neighborhoods. Use a combination of degree, neighbor
average degree, and a local search refinement to improve the greedy order.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_mis(adjacency, order):
    """Given a fixed order (list of vertex indices), greedily pick independent set.
    Returns the size of the independent set."""
    n = len(order)
    picked = np.zeros(n, dtype=bool)
    available = np.ones(n, dtype=bool)
    count = 0
    for v in order:
        if not available[v]:
            continue
        picked[v] = True
        count += 1
        # Mark neighbors as unavailable
        nbrs = np.where(adjacency[v])[0]
        available[nbrs] = False
    return count


def _score_vertices(adjacency):
    """Compute a score for each vertex. Higher score = picked earlier.
    
    We want to pick low-degree vertices first, but also consider the
    'value' of picking a vertex: it removes its neighbors from consideration.
    A good heuristic: score = -degree, but refined by considering the
    average degree of neighbors (picking a vertex with low-degree neighbors
    is better because it 'wastes' fewer potential picks).
    
    We use: score = -(degree + alpha * avg_neighbor_degree)
    where alpha is a tunable parameter.
    
    Additionally, we perform a local search refinement on the ordering.
    """
    n = adjacency.shape[0]
    degree = adjacency.sum(axis=1).astype(float)
    
    # Average degree of neighbors
    # For each vertex v, avg_nbr_deg[v] = sum(degree[u] for u in nbrs(v)) / degree[v]
    # We can compute this as: (adjacency @ degree) / degree, with care for degree=0
    nbr_deg_sum = adjacency @ degree
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_nbr_deg = np.where(degree > 0, nbr_deg_sum / degree, 0.0)
    
    # Score: lower is better for degree, so we negate
    # We want high score = pick early
    # Combine degree and neighbor structure
    alpha = 0.5
    raw_score = -(degree + alpha * avg_nbr_deg)
    
    # Initial order: sort by raw_score descending
    order = np.argsort(-raw_score)  # descending score
    
    # Local search refinement: try swapping adjacent elements in the order
    # and keep the swap if it improves the greedy MIS size
    current_size = _greedy_mis(adjacency, order)
    
    improved = True
    max_iters = 50
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(len(order) - 1):
            # Try swapping order[i] and order[i+1]
            order_new = order.copy()
            order_new[i], order_new[i+1] = order_new[i+1], order_new[i]
            new_size = _greedy_mis(adjacency, order_new)
            if new_size > current_size:
                order = order_new
                current_size = new_size
                improved = True
    
    # Convert the order back to scores: vertex that comes first gets highest score
    # We assign scores such that the greedy decoder (descending score) picks in this order
    scores = np.zeros(n, dtype=float)
    for rank, v in enumerate(order):
        scores[v] = -rank  # Higher score (less negative) for earlier picks
    
    return scores


def priority(adjacency):
    """Return one score per vertex. Vertices are selected greedily by descending score."""
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Check for timeout early
    if time.time() > deadline:
        # Fallback to simple degree-based scoring
        degree = adjacency.sum(axis=1).astype(float)
        return -degree
    
    try:
        scores = _score_vertices(adjacency)
    except Exception:
        # Fallback
        degree = adjacency.sum(axis=1).astype(float)
        scores = -degree
    
    return scores
# EVOLVE-BLOCK-END
