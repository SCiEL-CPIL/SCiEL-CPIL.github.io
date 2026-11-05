import numpy as np
import time

def priority(adjacency):
    """
    Assigns a priority score to each vertex for the greedy independent set decoder.
    Higher scores are picked first.
    
    Strategy:
    1. Initialize scores using a static heuristic based on structural features
       (degree, neighbor degree sum) to favor low-degree vertices far from hubs.
    2. Perform a local search (Simulated Annealing) to directly maximize the size 
       of the independent set produced by the greedy decoder.
    3. The local search simulates the greedy selection and updates vertex priorities 
       based on acceptance criteria that allow escaping local minima.
    """
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We aim to finish well before.
    # Let's use 140s as a hard stop for the search loop to ensure return.
    DEADLINE = 140.0
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
        
    # Ensure symmetry and remove self-loops for robustness
    adj = (adj | adj.T)
    np.fill_diagonal(adj, False)
    
    # Precompute neighbor lists as numpy arrays for fast vectorized deactivation
    neighbors_list = []
    for i in range(n):
        neighbors_list.append(np.where(adj[i])[0])
    
    # 1. Initial Heuristic
    deg = adj.sum(axis=1).astype(float)
    neighbor_deg_sum = adj @ deg
    # Avoid division by zero
    avg_neighbor_deg = neighbor_deg_sum / (deg + 1.0)
    
    # Initial scores: Higher is better.
    # We want low degree and low average neighbor degree.
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
    best_scores = scores.copy()
    
    rng = np.random.default_rng(42)
    
    # Annealing parameters
    # We use a time-based cooling schedule to ensure we utilize the full budget effectively.
    # T starts high to allow large jumps (worse solutions) and cools down.
    T_start = 5.0
    T_end = 0.01
    
    # We will run until the deadline.
    # To prevent infinite loops or excessive overhead, we don't pre-calculate steps,
    # but we check time frequently.
    
    while time.time() < DEADLINE:
        # Calculate current temperature based on progress
        elapsed = time.time() - start_time
        # Max expected time is DEADLINE - start_time (approx 140s)
        # We map elapsed to [0, 1]
        total_budget = DEADLINE - start_time
        if total_budget <= 0:
            break
        progress = min(1.0, elapsed / total_budget)
        T = T_start * (T_end / T_start) ** progress
        
        # Pick two random distinct vertices
        # Using integers for speed
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
                best_scores = current_scores.copy()
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
                
    return best_scores
