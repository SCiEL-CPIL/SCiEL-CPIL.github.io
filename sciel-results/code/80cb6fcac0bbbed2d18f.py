import numpy as np
import time

def _greedy_size(adj, order):
    """
    Simulate greedy MIS: pick vertices in 'order', skip neighbors of picked.
    Returns size of independent set.
    """
    n = len(order)
    picked = np.zeros(n, dtype=bool)
    size = 0
    for v in order:
        if not picked[v]:
            picked[v] = True
            size += 1
            nbrs = np.where(adj[v])[0]
            picked[nbrs] = True
    return size

def _get_order(scores):
    """
    Get the vertex order from scores (descending).
    """
    return np.argsort(-scores, kind='stable')

def _evaluate_order(adj, order):
    """
    Evaluate the greedy MIS size for a given order.
    """
    return _greedy_size(adj, order)

def _local_search(adj, scores, time_limit):
    """
    Efficient local search using adjacent swaps and position tracking.
    """
    start_time = time.time()
    n = len(scores)
    
    order = _get_order(scores)
    # Track positions: pos[v] = index of vertex v in order
    pos = np.empty(n, dtype=int)
    for idx, v in enumerate(order):
        pos[v] = idx
    
    current_size = _evaluate_order(adj, order)
    best_size = current_size
    best_scores = scores.copy()
    
    # Precompute neighbor lists as numpy arrays for fast access
    neighbor_lists = [np.where(adj[i])[0] for i in range(n)]
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        best_delta = 0
        best_swap_pos = None
        
        # Try adjacent swaps in the order (much faster than all pairs)
        for p in range(n - 1):
            if (time.time() - start_time) > time_limit:
                break
            
            v1 = order[p]
            v2 = order[p + 1]
            
            # Swap positions p and p+1
            new_order = order.copy()
            new_order[p], new_order[p + 1] = v2, v1
            
            new_size = _evaluate_order(adj, new_order)
            delta = new_size - current_size
            
            if delta > best_delta:
                best_delta = delta
                best_swap_pos = p
        
        if best_swap_pos is not None:
            p = best_swap_pos
            v1 = order[p]
            v2 = order[p + 1]
            
            # Apply swap to order
            order[p], order[p + 1] = v2, v1
            pos[v1], pos[v2] = p + 1, p
            
            # Update scores to match new order
            # We need to assign scores such that the order is preserved
            # Use rank-based scores
            for idx, v in enumerate(order):
                scores[v] = float(n - idx)
            
            current_size += best_delta
            improved = True
            
            if current_size > best_size:
                best_size = current_size
                best_scores = scores.copy()
        
        if (time.time() - start_time) > time_limit:
            break
    
    return best_scores, best_size

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy:
    1. Start with negative degree scores.
    2. Use efficient local search with adjacent swaps to optimize the vertex ordering.
    3. Return the scores that yield the best independent set size.
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s limit
    
    try:
        adj = np.asarray(adjacency, dtype=bool)
        n = adj.shape[0]
        
        if n < 2:
            return np.zeros(n)
        
        # Initial scores: negative degree (lower degree = higher priority)
        deg = adj.sum(axis=1)
        scores = -deg.astype(float).copy()
        
        # Run local search
        time_limit = DEADLINE - 5.0  # Leave 5s safety margin
        scores, best_size = _local_search(adj, scores, time_limit)
        
        if time.time() - start_time > DEADLINE:
            return -deg.astype(float)
            
        return scores
        
    except Exception:
        # Fallback to simple degree score
        try:
            deg = np.sum(adjacency, axis=1)
            return -deg.astype(float)
        except:
            return np.zeros(adjacency.shape[0])
