import numpy as np
import time

def priority(adjacency):
    """
    Assigns a priority score to each vertex for the greedy independent set decoder.
    Higher scores are picked first.
    
    Strategy: 
    1. Initialize scores using a static heuristic: Low Degree + Penalty for connecting to Hubs.
    2. Perform a local search (Simulated Annealing) to directly maximize the size 
       of the independent set produced by the greedy decoder.
    3. The local search simulates the greedy selection and updates vertex priorities 
       based on acceptance criteria that allow escaping local minima.
    """
    start_time = time.time()
    DEADLINE = 118.0  # Safety margin before 120s limit
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
        
    # Ensure symmetry and remove self-loops just in case, though input is likely clean
    # For boolean matrix, we can just use it as is if we trust the input, 
    # but let's be safe.
    # adj = (adj | adj.T)
    # np.fill_diagonal(adj, False)
    
    # Precompute neighbor lists as numpy arrays for fast vectorized deactivation
    neighbors_list = []
    for i in range(n):
        neighbors_list.append(np.where(adj[i])[0])
    
    # 1. Initial Heuristic
    deg = adj.sum(axis=1).astype(float)
    neighbor_deg_sum = adj @ deg
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
        order = np.argsort(-s, kind='stable')
        active = np.ones(n, dtype=bool)
        count = 0
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                if len(neighbors_list[v]) > 0:
                    active[neighbors_list[v]] = False
        return count

    # 3. Simulated Annealing Local Search
    best_size = simulate(scores)
    current_size = best_size
    current_scores = scores.copy()
    
    # Time budget: Use most of the available time
    max_time = DEADLINE - start_time
    if max_time < 1.0:
        return best_scores if 'best_scores' in locals() else scores
    
    rng = np.random.default_rng(42)
    
    # Annealing parameters
    T_start = 1.0
    T_end = 0.01
    
    # We will run until the deadline.
    # We use a time-based cooling schedule.
    
    while time.time() - start_time < DEADLINE:
        # Calculate current temperature
        elapsed = time.time() - start_time
        # Ensure we don't divide by zero or go negative
        progress = min(1.0, elapsed / max_time)
        T = T_start * (T_end / T_start) ** progress
        
        # Pick two random distinct vertices
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
            
        # Swap scores
        current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
        
        # Evaluate
        new_size = simulate(current_scores)
        
        delta = new_size - current_size
        
        if delta > 0:
            # Accept better move
            current_size = new_size
            if new_size > best_size:
                best_size = new_size
                scores = current_scores.copy()
        else:
            # Accept worse move with probability exp(delta / T)
            # delta is negative
            if T > 0:
                prob = np.exp(delta / T)
            else:
                prob = 0.0
                
            if rng.random() < prob:
                current_size = new_size
                # Do not update best_size if worse
            else:
                # Revert
                current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
                
    return scores
