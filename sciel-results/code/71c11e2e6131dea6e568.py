import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Precompute static features
    degrees = adj.sum(axis=1).astype(float)
    neighbor_deg_sum = adj @ degrees
    safe_deg = degrees + 1e-6
    mean_neighbor_deg = neighbor_deg_sum / safe_deg
    
    # 1. Initial Heuristic: Low Degree + Hub Avoidance
    # We want low degree vertices to have high priority (picked first).
    # Score = -degree - alpha * (mean neighbor degree)
    # This penalizes vertices connected to hubs.
    alpha = 0.5
    scores = -degrees - alpha * mean_neighbor_deg
    
    # Pre-compute neighbor lists for fast simulation
    neighbors_list = [np.where(adj[i])[0].tolist() for i in range(n)]
    
    def simulate(s):
        """
        Simulate greedy decoder with given scores.
        Returns size of independent set.
        """
        order = np.argsort(-s)
        active_set = set(range(n))
        count = 0
        for v in order:
            if v in active_set:
                count += 1
                active_set.remove(v)
                for u in neighbors_list[v]:
                    if u in active_set:
                        active_set.remove(u)
        return count
    
    # 2. Baseline
    best_size = simulate(scores)
    
    # 3. Local Search: Swap pairs
    # We will iterate until time limit.
    # To be efficient, we pick random pairs to swap.
    # If improvement found, keep swap. Else revert.
    
    max_time = 140.0 # Safety margin for 150s limit
    rng = np.random.default_rng(42)
    
    # We can also try to be smarter about which pairs to swap.
    # E.g., swap a vertex in the set with a vertex not in the set?
    # But random is simple and effective for N=200.
    
    while time.time() - start_time < max_time:
        # Pick two random vertices to swap
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
            
        # Swap scores
        scores[i], scores[j] = scores[j], scores[i]
        
        new_size = simulate(scores)
        
        if new_size > best_size:
            best_size = new_size
        else:
            # Revert
            scores[i], scores[j] = scores[j], scores[i]
            
    return scores
