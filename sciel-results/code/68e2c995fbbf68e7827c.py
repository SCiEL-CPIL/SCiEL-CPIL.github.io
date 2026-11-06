"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Assign a priority score to each vertex for a greedy Maximum Independent Set decoder.
    
    Strategy: Multi-stage local search with structural heuristics.
    1. Initialize with a composite score: -degree + bonus for low-degree vertices 
       that have high-degree neighbors (reducing effective blocking cost).
    2. Run targeted local search: try swapping/perturbing pairs of vertices.
    """
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 120s limit
    
    adj = np.asarray(adjacency)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    adj_bool = adj.astype(bool)
    degree = adj_bool.sum(axis=1).astype(float)
    
    # Pre-compute neighbor degree sums for the structural heuristic
    # For each vertex, sum of degrees of its neighbors
    neighbor_degree_sum = adj_bool @ degree  # vectorized
    
    # Composite initial score:
    # -degree: prefer low degree
    # + small * neighbor_degree_sum: prefer vertices whose neighbors are high-degree
    #   (selecting them blocks high-degree nodes, which is "cheap" in terms of lost options)
    scores = -degree + 0.1 * neighbor_degree_sum + np.random.normal(0, 0.5, n)
    
    # Fast greedy simulation
    def simulate_greedy(scores):
        selected = 0
        available = np.ones(n, dtype=bool)
        order = np.argsort(-scores, kind='stable')
        
        for v in order:
            if not available[v]:
                continue
            selected += 1
            available[v] = False
            nbrs = adj_bool[v]
            if nbrs.any():
                available[nbrs] = False
            if not available.any():
                break
        return selected
    
    current_size = simulate_greedy(scores)
    best_scores = scores.copy()
    best_size = current_size
    
    # Local search: try targeted perturbations
    # Phase 1: Random restarts with different initial heuristics
    heuristics = [
        lambda: -degree + np.random.normal(0, 1, n),
        lambda: -degree + 0.1 * neighbor_degree_sum + np.random.normal(0, 0.5, n),
        lambda: -np.sqrt(degree) + np.random.normal(0, 1, n),
        lambda: -np.log1p(degree) + np.random.normal(0, 1, n),
    ]
    
    for h in heuristics:
        if time.time() > deadline:
            break
        trial_scores = h()
        trial_size = simulate_greedy(trial_scores)
        if trial_size > best_size:
            best_size = trial_size
            best_scores = trial_scores.copy()
    
    # Phase 2: Local search on best_scores
    current_scores = best_scores.copy()
    current_size = best_size
    
    # Try pairwise swaps and single-vertex perturbations
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        improved = False
        
        # Try perturbing a random subset (5-15% of vertices)
        frac = np.random.uniform(0.05, 0.15)
        num_perturb = max(2, int(n * frac))
        indices = np.random.choice(n, num_perturb, replace=False)
        
        new_scores = current_scores.copy()
        # Mix of small and large perturbations
        small_noise = np.random.normal(0, 0.5, num_perturb)
        large_noise = np.random.normal(0, 2.0, num_perturb)
        # Randomly choose which get large vs small
        mask = np.random.random(num_perturb) < 0.3
        new_scores[indices] += np.where(mask, large_noise, small_noise)
        
        new_size = simulate_greedy(new_scores)
        
        if new_size > current_size:
            current_scores = new_scores
            current_size = new_size
            improved = True
            if current_size > best_size:
                best_size = current_size
                best_scores = current_scores.copy()
        
        # Try targeted: boost a random low-degree vertex, penalize a random high-degree one
        if not improved and np.random.random() < 0.5:
            low_deg_idx = np.where(degree < np.median(degree))[0]
            high_deg_idx = np.where(degree > np.median(degree))[0]
            if len(low_deg_idx) > 0 and len(high_deg_idx) > 0:
                v_boost = low_deg_idx[np.random.randint(len(low_deg_idx))]
                v_pen = high_deg_idx[np.random.randint(len(high_deg_idx))]
                new_scores2 = current_scores.copy()
                new_scores2[v_boost] += np.random.uniform(1.0, 3.0)
                new_scores2[v_pen] -= np.random.uniform(1.0, 3.0)
                new_size2 = simulate_greedy(new_scores2)
                if new_size2 > current_size:
                    current_scores = new_scores2
                    current_size = new_size2
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_scores = current_scores.copy()
        
        if not improved:
            no_improve_count += 1
        else:
            no_improve_count = 0
    
    return best_scores
# EVOLVE-BLOCK-END
