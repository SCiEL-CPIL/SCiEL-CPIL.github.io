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
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
        
    # Precompute neighbor lists as numpy arrays for fast vectorized deactivation
    neighbors_list = []
    for i in range(n):
        neighbors_list.append(np.where(adj[i])[0])
    
    # 1. Initial Heuristic
    deg = adj.sum(axis=1).astype(float)
    neighbor_deg_sum = adj @ deg
    avg_neighbor_deg = neighbor_deg_sum / (deg + 1.0)
    
    # Initial scores: Higher is better.
    scores = -deg - 0.5 * avg_neighbor_deg
    
    # 2. Fast Simulation Function
    def simulate(s):
        """
        Simulates the greedy decoder and returns the size of the independent set.
        """
        order = np.argsort(-s)
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
    
    # Time budget: 10 seconds for search to be safe (limit is 120s)
    max_time = 10.0
    rng = np.random.default_rng(42)
    
    # Annealing parameters
    T_start = 1.0
    T_end = 0.01
    # We don't know the number of steps in advance, so we can cool based on time or steps.
    # Let's estimate steps or just cool linearly with time remaining? 
    # Better: Cool exponentially with steps, but stop when time runs out.
    # Or simply: T = T_start * (T_end/T_start)^(elapsed/max_time)
    
    last_step_time = time.time()
    steps = 0
    
    while time.time() - start_time < max_time:
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
                
        steps += 1

    return scores
