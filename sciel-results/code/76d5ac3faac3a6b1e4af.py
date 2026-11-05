import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy:
    1. Initialize scores using a static heuristic (avg_neighbor_degree - degree).
    2. Convert scores to a vertex ordering.
    3. Use local search (hill climbing with random kicks) to improve the ordering.
       The objective is to maximize the size of the independent set produced by the greedy decoder.
    4. Return the scores corresponding to the best ordering found.
    """
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])

    # Compute degrees
    deg = adj.sum(axis=1).astype(float)
    
    # Compute sum of degrees of neighbors
    sum_deg_neighbors = adj.dot(deg)
    
    # Average degree of neighbors
    avg_deg_neighbors = np.zeros(n)
    mask = deg > 0
    avg_deg_neighbors[mask] = sum_deg_neighbors[mask] / deg[mask]
    
    # Initial scores: Favor low degree, high neighbor degree
    scores = avg_deg_neighbors - deg
    
    # Get initial ordering (descending score)
    order = np.argsort(-scores)
    
    def simulate(ord):
        """
        Runs the greedy decoder on the given vertex order.
        Returns the size of the independent set.
        """
        selected = np.zeros(n, dtype=bool)
        count = 0
        # Iterate through vertices in the given order
        for v in ord:
            if not selected[v]:
                count += 1
                # Mark v and all its neighbors as selected (blocked)
                selected[v] = True
                selected[adj[v, :]] = True
        return count

    # Evaluate initial solution
    best_order = order.copy()
    best_size = simulate(best_order)
    
    # Local Search
    start_time = time.time()
    deadline = start_time + 110.0 # Leave margin for safety
    
    # Use a random number generator
    rng = np.random.default_rng(42)
    
    stuck_count = 0
    max_stuck = 1000 # If no improvement for 1000 swaps, do a random kick
    
    while time.time() < deadline:
        # Pick two random distinct vertices to swap
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        while j == i:
            j = rng.integers(0, n)
        
        # Swap in a copy of the best order
        new_order = best_order.copy()
        new_order[i], new_order[j] = new_order[j], new_order[i]
        
        new_size = simulate(new_order)
        
        if new_size > best_size:
            best_size = new_size
            best_order = new_order
            stuck_count = 0
        else:
            stuck_count += 1
            if stuck_count > max_stuck:
                # Random kick: shuffle a few vertices to escape local minimum
                for _ in range(5):
                    k1 = rng.integers(0, n)
                    k2 = rng.integers(0, n)
                    if k1 != k2:
                        best_order[k1], best_order[k2] = best_order[k2], best_order[k1]
                # Re-evaluate after kick
                best_size = simulate(best_order)
                stuck_count = 0

    # Convert the best order back to scores
    # We want scores such that argsort(-scores) == best_order
    ranks = np.zeros(n, dtype=float)
    for idx, vertex in enumerate(best_order):
        ranks[vertex] = idx
    
    # Score = -rank ensures descending sort gives best_order
    final_scores = -ranks
    
    return final_scores
