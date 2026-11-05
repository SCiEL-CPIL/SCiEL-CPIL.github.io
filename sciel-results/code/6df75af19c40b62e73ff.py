"""Combined program: static heuristic init + fast local search for MIS greedy decoder."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _simulate_fast(nbrs, scores):
    """Fast greedy simulation: returns independent set size."""
    n = len(nbrs)
    if n == 0:
        return 0
    order = np.argsort(-scores, kind='stable')
    blocked = np.zeros(n, dtype=bool)
    count = 0
    for v in order:
        if not blocked[v]:
            count += 1
            blocked[v] = True
            if len(nbrs[v]) > 0:
                blocked[nbrs[v]] = True
    return count


def _simulate_fast_order(nbrs, order):
    """Fast greedy simulation given a precomputed order."""
    n = len(nbrs)
    if n == 0:
        return 0
    blocked = np.zeros(n, dtype=bool)
    count = 0
    for v in order:
        if not blocked[v]:
            count += 1
            blocked[v] = True
            if len(nbrs[v]) > 0:
                blocked[nbrs[v]] = True
    return count


def priority(adjacency):
    """
    Compute vertex scores for the Maximum Independent Set greedy decoder.
    
    Strategy:
    1. Initialize scores with a structural heuristic favoring low-degree vertices
       and penalizing vertices in dense neighborhoods.
    2. Refine scores via local search (random swaps) to directly maximize the
       independent set size produced by the greedy decoder.
    3. Use a deadline guard to ensure we return within time limits.
    """
    start_time = time.time()
    DEADLINE = start_time + 145.0  # Safety margin before 150s kill
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Precompute neighbor lists
    nbrs = [np.where(adj[i])[0] for i in range(n)]
    degree = np.array([len(nbrs[i]) for i in range(n)], dtype=float)
    
    # --- Initial Heuristic ---
    # Favor low degree vertices; penalize vertices with high neighbor degree sum
    # (i.e., vertices in dense regions or adjacent to hubs)
    
    # Mean neighbor degree
    mean_nd = np.zeros(n)
    for i in range(n):
        if degree[i] > 0:
            mean_nd[i] = np.mean(degree[nbrs[i]])
        else:
            mean_nd[i] = 0
    
    # Sum of neighbor degrees
    snd = np.zeros(n)
    for i in range(n):
        if degree[i] > 0:
            snd[i] = np.sum(degree[nbrs[i]])
    
    # Normalize features to [0, 1]
    def norm(x):
        x = np.asarray(x, dtype=float)
        min_val = x.min()
        max_val = x.max()
        if max_val == min_val:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)
    
    D_n = norm(degree)
    MeanND_n = norm(mean_nd)
    SND_n = norm(snd)
    
    # Initial scores: higher is better for the greedy decoder
    # We want low degree and low local density to be picked first
    # So score = -(w1 * degree + w2 * mean_neighbor_degree + w3 * sum_neighbor_degree)
    # Use weights that emphasize degree but account for local structure
    scores = -(0.6 * D_n + 0.3 * MeanND_n + 0.1 * SND_n)
    
    # --- Local Search: Random Swaps ---
    best_size = _simulate_fast(nbrs, scores)
    rng = np.random.default_rng(42)
    
    # Time budget for local search
    search_deadline = DEADLINE - 5.0  # Leave 5s safety margin
    
    # Adaptive step size: start with larger perturbations, reduce over time
    step_scale = 0.5
    
    while time.time() < search_deadline:
        # Pick two random distinct vertices
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        
        # Swap scores
        scores[i], scores[j] = scores[j], scores[i]
        
        # Evaluate
        new_size = _simulate_fast(nbrs, scores)
        
        if new_size > best_size:
            best_size = new_size
        else:
            # Revert
            scores[i], scores[j] = scores[j], scores[i]
        
        # Occasionally try a larger perturbation (swap more than one pair)
        if rng.random() < 0.05 and time.time() < search_deadline - 2.0:
            # Try swapping 3-5 random pairs
            num_swaps = rng.integers(3, 6)
            saved_scores = scores.copy()
            for _ in range(num_swaps):
                a, b = rng.integers(0, n, size=2)
                if a != b:
                    scores[a], scores[b] = scores[b], scores[a]
            
            new_size = _simulate_fast(nbrs, scores)
            if new_size <= best_size:
                scores[:] = saved_scores
    
    # --- Final Polish: Adjacent Swap Improvement ---
    # Try swapping adjacent pairs in the selection order for fine-tuning
    if time.time() < DEADLINE - 2.0:
        selected_order = np.argsort(-scores, kind='stable')
        
        for iteration in range(30):
            if time.time() > DEADLINE - 1.0:
                break
            
            improved = False
            # Try swapping adjacent pairs
            for i in range(len(selected_order) - 1):
                if time.time() > DEADLINE - 1.0:
                    break
                
                v1, v2 = selected_order[i], selected_order[i+1]
                selected_order[i], selected_order[i+1] = v2, v1
                
                new_size = _simulate_fast_order(nbrs, selected_order)
                
                if new_size > best_size:
                    best_size = new_size
                    improved = True
                else:
                    # Swap back
                    selected_order[i], selected_order[i+1] = v1, v2
            
            if not improved:
                break
        
        # Reconstruct scores from final order
        final_scores = np.zeros(n)
        for rank, v in enumerate(selected_order):
            final_scores[v] = -rank
        scores = final_scores
    
    return scores
# EVOLVE-BLOCK-END
