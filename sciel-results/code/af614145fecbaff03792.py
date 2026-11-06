<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Assign a priority score to each vertex for a greedy Maximum Independent Set decoder.
    
    Strategy: Stochastic Local Search (Simulated Annealing) on the score vector.
    We start with a good heuristic (inverse degree) and then randomly perturb scores,
    accepting changes that improve the greedy independent set size (with probability
    for worse moves to escape local minima).
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    adj = np.asarray(adjacency)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    adj_bool = adj.astype(bool)
    degree = adj_bool.sum(axis=1).astype(float)
    
    # Fast simulation of the greedy decoder
    def simulate_greedy_fast(scores):
        selected = 0
        available = np.ones(n, dtype=bool)
        
        # Get vertices in order of descending score
        order = np.argsort(-scores, kind='stable')
        
        for v in order:
            if not available[v]:
                continue
            # Select this vertex
            selected += 1
            # Remove it and its neighbors from available
            available[v] = False
            neighbors = np.where(adj_bool[v])[0]
            if len(neighbors) > 0:
                available[neighbors] = False
            
            # Early exit optimization
            if not available.any():
                break
        
        return selected

    # Initial score: Inverse degree (prefer low degree vertices)
    # Add a small random noise to break symmetries
    scores = -degree + np.random.normal(0, 1, n)
    current_size = simulate_greedy_fast(scores)
    
    best_scores = scores.copy()
    best_size = current_size
    
    # Simulated Annealing parameters
    T = 1.0
    cooling_rate = 0.999
    
    # Pre-compute neighbor lists for faster access? 
    # adj_bool[v] is already a boolean mask, which is fast for vectorized operations.
    
    while time.time() < deadline:
        # Perturb a random subset of scores
        # We perturb a fraction of the graph to allow large moves
        num_perturb = max(1, int(n * 0.1))
        indices = np.random.choice(n, num_perturb, replace=False)
        
        new_scores = scores.copy()
        # Add random noise
        noise = np.random.normal(0, 1, num_perturb)
        new_scores[indices] += noise
        
        new_size = simulate_greedy_fast(new_scores)
        
        delta = new_size - current_size
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta > 0:
            scores = new_scores
            current_size = new_size
            if current_size > best_size:
                best_size = current_size
                best_scores = scores.copy()
        else:
            # delta <= 0
            # If T is very small, probability is ~0
            if T > 1e-6:
                prob = np.exp(delta / T)
                if np.random.random() < prob:
                    scores = new_scores
                    current_size = new_size
        
        T *= cooling_rate
        
        # If T is too small, we might restart with a higher T if stuck?
        # For now, simple cooling.
        if T < 1e-4:
            break

    return best_scores
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Assign a priority score to each vertex for a greedy Maximum Independent Set decoder.
    
    Strategy: Fast greedy simulation with diverse initial heuristics, then local search
    to refine the best candidate.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    adj = np.asarray(adjacency)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    adj_bool = adj.astype(bool)
    degree = adj_bool.sum(axis=1).astype(float)
    
    # Pre-compute neighbor indices for each vertex for faster updates
    neighbors = [np.where(adj_bool[v])[0] for v in range(n)]
    
    # Fast simulation of the greedy decoder using list-based removal
    def simulate_greedy_fast(scores):
        selected = 0
        available = np.ones(n, dtype=bool)
        
        # Get vertices in order of descending score
        order = np.argsort(-scores, kind='stable')
        
        for v in order:
            if not available[v]:
                continue
            # Select this vertex
            selected += 1
            # Remove it and its neighbors from available
            available[v] = False
            nbrs = neighbors[v]
            if len(nbrs) > 0:
                available[nbrs] = False
            
            # Early exit optimization
            if not available.any():
                break
        
        return selected

    # Generate diverse initial score vectors
    candidates = []
    
    # 1. Inverse degree (prefer low degree)
    candidates.append(-degree + np.random.normal(0, 0.1, n))
    
    # 2. Negative degree squared (stronger preference for low degree)
    candidates.append(-degree**2 + np.random.normal(0, 1, n))
    
    # 3. Random
    candidates.append(np.random.normal(0, 1, n))
    
    # 4. Based on clustering coefficient proxy (local density)
    # Approximate by looking at degree of neighbors
    local_density = np.zeros(n)
    for v in range(n):
        nbrs = neighbors[v]
        if len(nbrs) > 0:
            local_density[v] = degree[nbrs].mean()
    candidates.append(-local_density + np.random.normal(0, 0.1, n))
    
    # 5. Degree ratio (degree / mean_degree)
    mean_deg = degree.mean()
    if mean_deg > 0:
        candidates.append(-degree / mean_deg + np.random.normal(0, 0.1, n))
    
    # Evaluate all candidates and pick the best
    best_scores = None
    best_size = 0
    current_scores = None
    current_size = 0
    
    for cand in candidates:
        size = simulate_greedy_fast(cand)
        if size > best_size:
            best_size = size
            best_scores = cand.copy()
            current_scores = cand.copy()
            current_size = size
    
    # Local search refinement on the best candidate
    T = 1.0
    cooling_rate = 0.9995
    stuck_count = 0
    max_stuck = 200
    
    while time.time() < deadline:
        # Check if we're stuck
        if stuck_count > max_stuck:
            # Restart with a different initial candidate
            for cand in candidates:
                size = simulate_greedy_fast(cand)
                if size > best_size:
                    best_size = size
                    best_scores = cand.copy()
                    current_scores = cand.copy()
                    current_size = size
                    break
            else:
                # Use random restart
                current_scores = np.random.normal(0, 1, n)
                current_size = simulate_greedy_fast(current_scores)
            T = 1.0
            stuck_count = 0
            max_stuck = int(max_stuck * 1.5)
            continue
        
        # Perturb a random subset of scores
        num_perturb = max(1, int(n * 0.05))
        indices = np.random.choice(n, num_perturb, replace=False)
        
        new_scores = current_scores.copy()
        noise = np.random.normal(0, 0.5, num_perturb)
        new_scores[indices] += noise
        
        new_size = simulate_greedy_fast(new_scores)
        
        delta = new_size - current_size
        
        if delta > 0:
            current_scores = new_scores
            current_size = new_size
            if current_size > best_size:
                best_size = current_size
                best_scores = current_scores.copy()
            stuck_count = 0
        else:
            if T > 1e-6:
                prob = np.exp(delta / T)
                if np.random.random() < prob:
                    current_scores = new_scores
                    current_size = new_size
            stuck_count += 1
        
        T *= cooling_rate
        
        if T < 1e-4:
            T = 0.5
            stuck_count = 0

    return best_scores
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
