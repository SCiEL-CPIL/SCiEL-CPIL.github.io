import numpy as np
import time
import random

def priority(adjacency):
    """
    Computes a priority score for each vertex to maximize the size of the
    independent set selected by a greedy decoder.
    """
    adj = np.asarray(adjacency)
    n = adj.shape[0]
    
    # Precompute neighbor lists for fast access
    # neighbors[i] is a list of indices adjacent to i
    neighbors = [list(np.where(adj[i] == 1)[0]) for i in range(n)]
    
    # Helper to evaluate the MIS size for a given ordering
    def eval_ordering(order):
        visited = [False] * n
        count = 0
        for v in order:
            if not visited[v]:
                count += 1
                for u in neighbors[v]:
                    visited[u] = True
        return count

    # Initial ordering: Heuristic based on degree (lowest degree first)
    # This is a strong baseline for MIS
    degrees = adj.sum(axis=1)
    # Sort indices by degree ascending
    initial_order = list(np.argsort(degrees))
    
    current_order = initial_order
    current_score = eval_ordering(current_order)
    best_order = current_order.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    # We have ~120s limit. 5 graphs. ~20s per graph.
    # Let's use a time-based loop.
    start_time = time.time()
    deadline = start_time + 18.0  # Leave 2s margin for 5 graphs (18*5=90s)
    
    T = 1.0  # Initial temperature
    T_min = 1e-4
    cooling = 0.995
    
    # Random seed for reproducibility in development, but usually random is fine
    # random.seed(42) 
    
    while time.time() < deadline:
        # Generate neighbor state: swap two random positions
        i, j = random.sample(range(n), 2)
        new_order = current_order.copy()
        new_order[i], new_order[j] = new_order[j], new_order[i]
        
        new_score = eval_ordering(new_order)
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < np.exp(delta / T):
            current_order = new_order
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_order = current_order.copy()
        
        T = max(T * cooling, T_min)
        
        # If T is very low and we are stuck, we might restart or just continue
        # Given the small N, local minima are common. 
        # A simple restart strategy: if no improvement for a while, 
        # we could re-initialize, but let's keep it simple first.
        
    # Convert best_order to scores
    # Higher score = selected earlier
    # If best_order is [v0, v1, ...], v0 should have highest score.
    scores = np.zeros(n)
    for idx, v in enumerate(best_order):
        scores[v] = n - idx
        
    return scores.astype(float)
