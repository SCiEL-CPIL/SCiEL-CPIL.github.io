# EVOLVE-BLOCK-START
import numpy as np
import time

def _greedy_size(order, adj):
    n = len(order)
    selected = 0
    blocked = np.zeros(n, dtype=bool)
    for v in order:
        if not blocked[v]:
            selected += 1
            blocked[v] = True
            blocked |= adj[v]
    return selected

def priority(adjacency):
    n = adjacency.shape[0]
    if n == 0:
        return np.array([], dtype=float)
        
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # Initial Score: Min Degree + Min Neighbor Degree Sum
    degrees = adjacency.sum(axis=1)
    neighbor_deg_sum = adjacency @ degrees
    
    # We want to pick low degree, low neighbor degree sum first.
    # Score = - (degree + 0.5 * neighbor_deg_sum)
    # Higher score is picked first.
    base_score = -(degrees + 0.5 * neighbor_deg_sum)
    
    # Initial order: descending score
    order = np.argsort(base_score)[::-1]
    
    current_size = _greedy_size(order, adjacency)
    best_order = order.copy()
    best_size = current_size
    
    # Local Search
    rng = np.random.default_rng()
    
    # Track time for logging/debugging if needed, but we just loop
    while time.time() < deadline:
        # Random swap
        i, j = rng.integers(0, n, 2)
        if i == j:
            continue
            
        order[i], order[j] = order[j], order[i]
        new_size = _greedy_size(order, adjacency)
        
        if new_size > best_size:
            best_size = new_size
            best_order = order.copy()
        else:
            # Revert
            order[i], order[j] = order[j], order[i]
            
    # Convert best_order to scores
    # Score[v] should be such that sorting descending gives best_order
    # rank(v) = index in best_order
    # score = -rank
    ranks = np.empty(n, dtype=int)
    for idx, v in enumerate(best_order):
        ranks[v] = idx
        
    scores = -ranks.astype(float)
    return scores
# EVOLVE-BLOCK-END
