import numpy as np
import time

def priority(adjacency):
    """
    Assigns a priority score to each vertex for the greedy independent set decoder.
    Higher scores are picked first.
    
    Strategy: 
    1. Initialize scores using a static heuristic: Low Degree + Penalty for connecting to Hubs.
    2. Perform a local search (swap-based optimization) to directly maximize the size 
       of the independent set produced by the greedy decoder.
    3. The local search simulates the greedy selection and swaps vertex priorities 
       if it leads to a larger independent set.
    """
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
        
    # Precompute neighbor lists as numpy arrays for fast vectorized deactivation
    # adj is n x n boolean matrix
    neighbors_list = []
    for i in range(n):
        neighbors_list.append(np.where(adj[i])[0])
    
    # 1. Initial Heuristic
    # We want low degree vertices to have high scores (picked early).
    # We also penalize vertices connected to high-degree vertices (hubs), 
    # because picking a neighbor of a hub "wastes" the hub's blocking potential 
    # or blocks a hub that could have blocked many others if picked itself? 
    # Actually, standard MIS heuristics often prefer low degree.
    # Let's use: Score = -degree - alpha * (sum of neighbor degrees) / (degree + 1)
    # The second term penalizes vertices that are "surrounded" by high-degree nodes,
    # which might be less "independent" in spirit, or rather, it's a refinement 
    # to break ties and guide the initial state closer to a good solution.
    
    deg = adj.sum(axis=1).astype(float)
    # Sum of degrees of neighbors
    neighbor_deg_sum = adj @ deg
    # Average degree of neighbors
    avg_neighbor_deg = neighbor_deg_sum / (deg + 1.0)
    
    # Initial scores: Higher is better.
    # We want to pick low degree vertices.
    # So base score is -deg.
    # We subtract a penalty for being near hubs? 
    # If a vertex is near a hub, it's "crowded".
    # Let's try: Score = -deg - 0.5 * avg_neighbor_deg
    scores = -deg - 0.5 * avg_neighbor_deg
    
    # 2. Fast Simulation Function
    def simulate(s):
        """
        Simulates the greedy decoder and returns the size of the independent set.
        """
        order = np.argsort(-s) # Descending score
        active = np.ones(n, dtype=bool)
        count = 0
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                # Deactivate neighbors
                if len(neighbors_list[v]) > 0:
                    active[neighbors_list[v]] = False
        return count

    # 3. Local Search
    best_size = simulate(scores)
    
    # Time budget: 10 seconds for search to be safe (limit is 120s, but overhead exists)
    max_time = 10.0
    rng = np.random.default_rng(42)
    
    # We will perform random swaps.
    # To improve convergence, we can bias swaps towards vertices that are close in rank.
    # However, random swaps over the whole set often work well for N=200.
    
    while time.time() - start_time < max_time:
        # Pick two random distinct vertices
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
            
        # Swap scores
        scores[i], scores[j] = scores[j], scores[i]
        
        # Evaluate
        new_size = simulate(scores)
        
        if new_size > best_size:
            best_size = new_size
        else:
            # Revert
            scores[i], scores[j] = scores[j], scores[i]
            
        # Optional: Early stop if we hit a theoretical upper bound?
        # A simple upper bound is N / (avg_degree + 1) is not tight.
        # We can just run until time limit.
        
    return scores
