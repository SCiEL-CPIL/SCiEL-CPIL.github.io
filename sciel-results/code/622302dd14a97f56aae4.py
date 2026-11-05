import numpy as np
import time
import math

def priority(adjacency):
    """
    Assigns a priority score to each vertex for the greedy independent set decoder.
    Higher scores are picked first.
    
    Strategy: 
    1. Initialize scores using a strong static heuristic: Low Degree + Penalty for connecting to Hubs.
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
    
    # Time budget: Use most of the available time, but stop before DEADLINE
    # We check time inside the loop.
    rng = np.random.default_rng(42)
    
    # Annealing parameters
    # We need to estimate how many steps we can take to tune T.
    # Let's just use a time-based cooling or a step-based cooling with a large step count.
    # Since we don't know the exact number of steps, time-based cooling is safer.
    # T(t) = T_start * (T_end/T_start)^(t / T_total)
    # Let's assume we run for the full duration.
    
    T_start = 1.0
    T_end = 0.01
    total_time_budget = DEADLINE - start_time
    
    # We will run until DEADLINE - 2.0 seconds to be safe.
    stop_time = start_time + total_time_budget - 2.0
    
    while time.time() < stop_time:
        # Calculate current temperature
        elapsed = time.time() - start_time
        # Ensure progress is between 0 and 1
        progress = min(1.0, elapsed / total_time_budget)
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
                # Avoid overflow/underflow issues if T is very small and delta is large negative
                # But delta is at most -n, T is >= 0.01. exp(-200/0.01) is 0.
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
