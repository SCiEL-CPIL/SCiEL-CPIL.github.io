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
    
    # Fast greedy simulation using bitset-like approach with numpy
    def simulate_greedy_fast(scores):
        """Fast greedy: process vertices in score order, skip if already blocked."""
        selected = 0
        blocked = np.zeros(n, dtype=bool)
        order = np.argsort(-scores, kind='stable')
        
        for v in order:
            if blocked[v]:
                continue
            selected += 1
            # Block v and all its neighbors
            blocked[v] = True
            nbrs = adj_bool[v]
            if nbrs.any():
                blocked[nbrs] = True
            if not (~blocked).any():
                break
        return selected
    
    def simulate_greedy(scores):
        """Alias for fast simulation."""
        return simulate_greedy_fast(scores)
    
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
    
    # Phase 2: Enhanced local search with diverse move types
    current_scores = best_scores.copy()
    current_size = best_size
    
    no_improve_count = 0
    max_no_improve = 80
    
    # Precompute median degree for targeted moves
    med_deg = np.median(degree)
    low_deg_mask = degree < med_deg
    high_deg_mask = degree > med_deg
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        improved = False
        move_type = np.random.choice(4)  # 0: subset perturb, 1: targeted, 2: swap, 3: re-rank
        
        if move_type == 0:
            # Perturb a random subset with varying noise scales
            frac = np.random.uniform(0.03, 0.20)
            num_perturb = max(2, int(n * frac))
            indices = np.random.choice(n, num_perturb, replace=False)
            
            new_scores = current_scores.copy()
            # Adaptive noise: smaller for high-degree, larger for low-degree
            noise_scale = 1.0 / (1.0 + 0.1 * degree[indices])
            noise = np.random.normal(0, 1.0, num_perturb) * noise_scale
            # Occasionally add a big kick to one vertex
            if np.random.random() < 0.3:
                kick_idx = np.random.randint(num_perturb)
                noise[kick_idx] += np.random.choice([-1, 1]) * np.random.uniform(2.0, 5.0)
            new_scores[indices] += noise
            
            new_size = simulate_greedy(new_scores)
            
            if new_size > current_size:
                current_scores = new_scores
                current_size = new_size
                improved = True
                if current_size > best_size:
                    best_size = current_size
                    best_scores = current_scores.copy()
        
        elif move_type == 1:
            # Targeted: boost low-degree, penalize high-degree
            low_indices = np.where(low_deg_mask)[0]
            high_indices = np.where(high_deg_mask)[0]
            if len(low_indices) > 0 and len(high_indices) > 0:
                num_boost = max(1, int(np.random.uniform(0.01, 0.05) * n))
                num_pen = max(1, int(np.random.uniform(0.01, 0.05) * n))
                
                boost_idx = np.random.choice(low_indices, min(num_boost, len(low_indices)), replace=False)
                pen_idx = np.random.choice(high_indices, min(num_pen, len(high_indices)), replace=False)
                
                new_scores = current_scores.copy()
                new_scores[boost_idx] += np.random.uniform(0.5, 3.0, len(boost_idx))
                new_scores[pen_idx] -= np.random.uniform(0.5, 3.0, len(pen_idx))
                
                new_size = simulate_greedy(new_scores)
                if new_size > current_size:
                    current_scores = new_scores
                    current_size = new_size
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_scores = current_scores.copy()
        
        elif move_type == 2:
            # Swap scores between two random vertices
            i, j = np.random.choice(n, 2, replace=False)
            new_scores = current_scores.copy()
            new_scores[i], new_scores[j] = new_scores[j], new_scores[i]
            
            new_size = simulate_greedy(new_scores)
            if new_size > current_size:
                current_scores = new_scores
                current_size = new_size
                improved = True
                if current_size > best_size:
                    best_size = current_size
                    best_scores = current_scores.copy()
        
        else:
            # Re-rank: add small random noise to all scores, then re-sort effect
            # This simulates a different tie-breaking
            new_scores = current_scores + np.random.normal(0, 0.1, n)
            new_size = simulate_greedy(new_scores)
            if new_size > current_size:
                current_scores = new_scores
                current_size = new_size
                improved = True
                if current_size > best_size:
                    best_size = current_size
                    best_scores = current_scores.copy()
        
        if not improved:
            no_improve_count += 1
        else:
            no_improve_count = 0
    
    # Phase 3: Final polish pass - fine-grained adjustments on best solution
    if time.time() < deadline - 5:
        polish_scores = best_scores.copy()
        polish_size = best_size
        polish_no_improve = 0
        max_polish_no_improve = 30
        
        while time.time() < deadline - 5 and polish_no_improve < max_polish_no_improve:
            # Try single-vertex fine adjustments
            v = np.random.randint(n)
            delta = np.random.normal(0, 0.3)
            
            new_scores = polish_scores.copy()
            new_scores[v] += delta
            
            new_size = simulate_greedy(new_scores)
            if new_size > polish_size:
                polish_scores = new_scores
                polish_size = new_size
                polish_no_improve = 0
                if polish_size > best_size:
                    best_size = polish_size
                    best_scores = polish_scores.copy()
            else:
                polish_no_improve += 1
    
    return best_scores
# EVOLVE-BLOCK-END
