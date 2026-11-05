import numpy as np
import time

def priority(adjacency):
    """
    Assigns a priority score to each vertex for the greedy independent set decoder.
    Higher scores are picked first.
    
    Strategy: 
    1. Initialize scores using a static heuristic: Low Degree + Penalty for connecting to Hubs.
    2. Perform Simulated Annealing to maximize independent set size via greedy decoder.
    3. Final polish phase with focused local search on best solution found.
    """
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. Use 140s as hard stop.
    DEADLINE = 140.0
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
        
    # Precompute neighbor lists and degrees
    neighbors_list = [np.where(adj[i])[0] for i in range(n)]
    deg = adj.sum(axis=1).astype(float)
    
    # 1. Initial Heuristic
    neighbor_deg_sum = adj @ deg
    avg_neighbor_deg = neighbor_deg_sum / (deg + 1.0)
    scores = -deg - 0.5 * avg_neighbor_deg
    
    # 2. Fast Simulation Function
    def simulate(s):
        """Simulates greedy decoder, returns independent set size."""
        order = np.argsort(-s)
        active = np.ones(n, dtype=bool)
        count = 0
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                nbrs = neighbors_list[v]
                if len(nbrs) > 0:
                    active[nbrs] = False
        return count

    # 3. Simulated Annealing
    best_size = simulate(scores)
    current_size = best_size
    current_scores = scores.copy()
    rng = np.random.default_rng(42)
    
    T_start = 1.0
    T_end = 0.001
    
    # Phase 1: Main SA (use ~80% of budget)
    sa_end_time = start_time + (DEADLINE - start_time) * 0.8
    
    while time.time() < sa_end_time:
        elapsed = time.time() - start_time
        total_budget = sa_end_time - start_time
        progress = min(1.0, elapsed / total_budget)
        T = T_start * (T_end / T_start) ** progress
        
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
            
        current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
        new_size = simulate(current_scores)
        delta = new_size - current_size
        
        if delta > 0:
            current_size = new_size
            if new_size > best_size:
                best_size = new_size
                scores = current_scores.copy()
        else:
            if T > 1e-10:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_size = new_size
                else:
                    current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
            else:
                current_scores[i], current_scores[j] = current_scores[j], current_scores[i]

    # Phase 2: Final polish - focused local search on best solution
    polish_end_time = DEADLINE - 5.0  # Leave 5s safety margin
    current_scores = scores.copy()
    current_size = best_size
    
    while time.time() < polish_end_time:
        # Try swapping pairs of vertices to improve
        improved = False
        # Sample random pairs to try
        for _ in range(50):
            i, j = rng.integers(0, n, size=2)
            if i == j:
                continue
            current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
            new_size = simulate(current_scores)
            if new_size > current_size:
                current_size = new_size
                improved = True
                if new_size > best_size:
                    best_size = new_size
                    scores = current_scores.copy()
            else:
                current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
        if not improved:
            break
    
    return scores
