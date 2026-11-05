import numpy as np
import time

def priority(adjacency):
    """
    Calculate priority scores for a greedy Maximum Independent Set decoder.
    
    Strategy:
    1. Model vertex priority as solution to MWIS on complement graph, where weights
       derive from local density of original graph's neighborhood. Vertices in isolated
       clusters (dense in complement) get higher priority.
    2. Initialize scores using these complement-density weights.
    3. Refine via local search (pair swaps) to maximize greedy independent set size.
    4. Final polish pass with tighter tolerance.
    """
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s limit
    
    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]
    
    if N == 0:
        return np.array([])
    
    # Ensure symmetry and remove self-loops
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    
    # Convert to boolean for faster neighbor lookups
    A_bool = A > 0.5
    
    # Precompute neighbor lists as numpy arrays for vectorized deactivation
    neighbors_list = [np.where(A_bool[i])[0] for i in range(N)]
    
    # --- 1. Complement-density based weights ---
    # The complement graph has edge (i,j) iff original has NO edge.
    # MWIS on complement = maximum clique in complement = maximum independent set in original.
    # Weight vertices by how "isolated" they are in the original graph.
    
    deg = A.sum(axis=1)
    
    # Sum of degrees of neighbors (local density proxy)
    neighbor_deg_sum = A.dot(deg)
    
    # Number of edges among neighbors (local clustering)
    # For each vertex i, count edges among its neighbors
    local_edge_counts = np.zeros(N)
    for i in range(N):
        nbrs = neighbors_list[i]
        if len(nbrs) > 1:
            # Count edges in the subgraph induced by nbrs
            sub = A[np.ix_(nbrs, nbrs)]
            local_edge_counts[i] = sub.sum() / 2.0
    
    # Local density: ratio of edges among neighbors to possible edges
    possible_edges = np.maximum(deg * (deg - 1) / 2.0, 1e-6)
    local_density = local_edge_counts / possible_edges
    
    # We want LOW local density (isolated clusters) and LOW degree
    # Weight = inverse of local density, normalized
    def normalize(x):
        if x.max() > x.min():
            return (x - x.min()) / (x.max() - x.min())
        else:
            return np.zeros_like(x)
            
    norm_deg = normalize(deg)
    norm_local_density = normalize(local_density)
    
    # Complement-density weight: high for isolated vertices
    # Combine: low degree + low local density
    weights = -(0.6 * norm_deg + 0.4 * norm_local_density)
    weights = normalize(weights)
    
    def simulate_fast(s):
        """
        Fast greedy simulation.
        Returns the size of the independent set.
        Uses vectorized numpy operations for deactivation.
        """
        order = np.argsort(-s, kind='stable')
        active = np.ones(N, dtype=bool)
        count = 0
        
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                nbrs = neighbors_list[v]
                if len(nbrs) > 0:
                    # Vectorized deactivation of neighbors
                    active[nbrs] = False
        return count
    
    # --- 2. Baseline ---
    best_val = simulate_fast(weights)
    best_scores = weights.copy()
    
    # --- 3. Local Search ---
    rng = np.random.default_rng(42)
    
    # Precompute score order for structured swaps
    score_order = np.argsort(-weights)
    
    # Try multiple restarts
    num_restarts = 15
    for restart in range(num_restarts):
        if time.time() - start_time > DEADLINE:
            break
        
        # For restart 0, use original. For others, perturb from best found.
        if restart == 0:
            current_scores = best_scores.copy()
            current_val = best_val
        else:
            # Add noise to escape local optima
            noise_scale = 0.1
            noise = rng.normal(0, noise_scale, N)
            current_scores = best_scores + noise
            current_val = simulate_fast(current_scores)
        
        # Local search loop
        no_improve = 0
        # Allow more iterations since simulation is faster
        max_no_improve = 5000 + 1000 * restart
        
        while time.time() - start_time < DEADLINE and no_improve < max_no_improve:
            # Choose swap candidates
            # 60% structured (close in rank), 40% random
            if rng.random() < 0.6:
                base_idx = rng.integers(0, N - 1)
                # Pick a nearby index
                offset = rng.integers(1, min(20, N - 1))
                i = score_order[base_idx]
                j = score_order[min(base_idx + offset, N - 1)]
            else:
                i, j = rng.integers(0, N, size=2)
            
            if i == j:
                continue
            
            # Swap scores
            current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
            new_val = simulate_fast(current_scores)
            
            if new_val > current_val:
                current_val = new_val
                no_improve = 0
                if current_val > best_val:
                    best_val = current_val
                    best_scores = current_scores.copy()
                    # Update score order if we find a new best
                    score_order = np.argsort(-current_scores)
            else:
                # Revert swap
                current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
                no_improve += 1
        
        # Early stopping if no improvement in this restart
        if current_val <= best_val and restart > 2:
            break
    
    # --- 4. Final polish pass with tighter tolerance ---
    if time.time() - start_time < DEADLINE - 5.0:
        polish_scores = best_scores.copy()
        polish_val = best_val
        polish_order = np.argsort(-polish_scores)
        
        # More focused local search with smaller perturbations
        for _ in range(2000):
            if time.time() - start_time > DEADLINE - 3.0:
                break
            
            # Smaller random swaps
            i, j = rng.integers(0, N, size=2)
            if i == j:
                continue
            
            polish_scores[i], polish_scores[j] = polish_scores[j], polish_scores[i]
            new_val = simulate_fast(polish_scores)
            
            if new_val > polish_val:
                polish_val = new_val
                if polish_val > best_val:
                    best_val = polish_val
                    best_scores = polish_scores.copy()
            else:
                # Revert
                polish_scores[i], polish_scores[j] = polish_scores[j], polish_scores[i]
    
    return best_scores
